"""
Test that 2-kernel broadphase (delayed validation) produces identical
contacts to single-kernel SAP broadphase.

Uses the G1 humanoid fall scenario from the rigid benchmarks with random
forces applied across 4096 parallel environments.
"""

import numpy as np
import pytest
import torch

import genesis as gs

from .utils import get_hf_dataset


def _create_g1_fall_scene(n_envs, broadphase_kernels):
    """Build a G1 humanoid falling scene with the specified broadphase kernel count."""
    scene = gs.Scene(
        rigid_options=gs.options.RigidOptions(
            dt=0.005,
            iterations=10,
            tolerance=1e-5,
            ls_iterations=20,
            constraint_solver=gs.constraint_solver.Newton,
            broadphase_kernels=broadphase_kernels,
        ),
        show_viewer=False,
        show_FPS=False,
    )

    scene.add_entity(gs.morphs.Plane())

    asset_path = get_hf_dataset(pattern="unitree_g1/*")
    robot = scene.add_entity(
        gs.morphs.MJCF(
            file=f"{asset_path}/unitree_g1/g1_29dof_rev_1_0.xml",
            pos=(0, 0, 1.0),
        ),
        vis_mode="collision",
    )

    scene.build(n_envs=n_envs)

    init_qpos = torch.zeros((robot.n_qs,), dtype=gs.tc_float, device=gs.device)
    init_qpos[2] = 1.0
    init_qpos[3] = 1.0
    robot.set_qpos(init_qpos)

    return scene, robot


def _sort_contacts(contacts, n_envs):
    """Return per-env sort indices that order contacts by (geom_a, geom_b, |penetration|).

    Invalid entries (geom_a == -1) are pushed to the end.
    """
    ga = contacts["geom_a"].to(torch.float64)
    gb = contacts["geom_b"].to(torch.float64)
    pen = contacts["penetration"].to(torch.float64)

    key = ga * 1e10 + gb * 1e5 + pen.abs()
    key = torch.where(contacts["geom_a"] >= 0, key, 1e15)

    _, order = key.sort(dim=1, stable=True)
    return order


@pytest.mark.slow
@pytest.mark.parametrize("backend", [gs.gpu])
def test_broadphase_two_kernel_vs_single_kernel(backend):
    """2-kernel broadphase must produce the same contacts as single-kernel SAP."""
    n_envs = 4096
    n_steps = 30
    max_force = 50.0

    scene_1k, robot_1k = _create_g1_fall_scene(n_envs, broadphase_kernels=1)
    scene_2k, robot_2k = _create_g1_fall_scene(n_envs, broadphase_kernels=2)

    torch.manual_seed(42)
    forces = torch.zeros((n_envs, robot_1k.n_dofs), dtype=gs.tc_float, device=gs.device)

    for step in range(n_steps):
        forces.uniform_(-max_force, max_force)
        robot_1k.control_dofs_force(forces)
        robot_2k.control_dofs_force(forces)

        scene_1k.step()
        scene_2k.step()

        # --- contact count must match per environment ---
        n_contacts_1k = scene_1k.rigid_solver.collider._collider_state.n_contacts.to_numpy()
        n_contacts_2k = scene_2k.rigid_solver.collider._collider_state.n_contacts.to_numpy()
        np.testing.assert_array_equal(
            n_contacts_1k,
            n_contacts_2k,
            err_msg=f"Step {step}: per-env contact count mismatch",
        )

        max_contacts = int(n_contacts_1k.max())
        if max_contacts == 0:
            continue

        # --- contact data must match (order may differ due to atomics) ---
        contacts_1k = scene_1k.rigid_solver.collider.get_contacts(as_tensor=True, to_torch=True)
        contacts_2k = scene_2k.rigid_solver.collider.get_contacts(as_tensor=True, to_torch=True)

        order_1k = _sort_contacts(contacts_1k, n_envs)
        order_2k = _sort_contacts(contacts_2k, n_envs)

        for field in ("geom_a", "geom_b", "link_a", "link_b"):
            sorted_1k = contacts_1k[field].gather(1, order_1k)
            sorted_2k = contacts_2k[field].gather(1, order_2k)
            assert torch.equal(sorted_1k, sorted_2k), f"Step {step}: {field} mismatch"

        sorted_pen_1k = contacts_1k["penetration"].gather(1, order_1k)
        sorted_pen_2k = contacts_2k["penetration"].gather(1, order_2k)
        assert torch.equal(sorted_pen_1k, sorted_pen_2k), f"Step {step}: penetration mismatch"

        for field in ("position", "normal"):
            data_1k = contacts_1k[field]
            data_2k = contacts_2k[field]
            idx_1k = order_1k.unsqueeze(-1).expand_as(data_1k)
            idx_2k = order_2k.unsqueeze(-1).expand_as(data_2k)
            sorted_1k = data_1k.gather(1, idx_1k)
            sorted_2k = data_2k.gather(1, idx_2k)
            assert torch.equal(sorted_1k, sorted_2k), f"Step {step}: {field} mismatch"
