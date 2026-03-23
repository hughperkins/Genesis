"""
Test that 2-kernel broadphase (delayed validation) produces identical
collision pairs to single-kernel SAP broadphase.

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


def _get_sorted_broad_pairs(scene):
    """Extract broadphase collision pairs and sort them per-env by (geom_a, geom_b).

    Returns (n_broad_pairs, sorted_pairs) where:
      - n_broad_pairs: numpy array of shape (n_envs,) with pair counts
      - sorted_pairs:  numpy array of shape (n_envs, max_pairs, 2) with sorted geom ID pairs
    """
    state = scene.rigid_solver.collider._collider_state
    n_pairs = state.n_broad_pairs.to_numpy()
    max_pairs = int(n_pairs.max()) if n_pairs.max() > 0 else 0

    if max_pairs == 0:
        return n_pairs, None

    raw = state.broad_collision_pairs.to_numpy()[:max_pairs]  # (max_pairs, n_envs, 2)
    pairs = raw.transpose(1, 0, 2)  # (n_envs, max_pairs, 2)

    n_envs = pairs.shape[0]
    sorted_pairs = np.empty_like(pairs)
    for i_b in range(n_envs):
        n = n_pairs[i_b]
        if n == 0:
            sorted_pairs[i_b] = pairs[i_b]
            continue
        valid = pairs[i_b, :n]
        order = np.lexsort((valid[:, 1], valid[:, 0]))
        sorted_pairs[i_b, :n] = valid[order]
        sorted_pairs[i_b, n:] = pairs[i_b, n:]

    return n_pairs, sorted_pairs


@pytest.mark.slow
@pytest.mark.parametrize("backend", [gs.gpu])
def test_broadphase_two_kernel_vs_single_kernel(backend):
    """2-kernel broadphase must produce the same collision pairs as single-kernel SAP."""
    n_envs = 4096
    n_steps = 30
    max_force = 50.0

    scene_1k, robot_1k = _create_g1_fall_scene(n_envs, broadphase_kernels=1)
    scene_2k, robot_2k = _create_g1_fall_scene(n_envs, broadphase_kernels=2)

    torch.manual_seed(42)
    forces = torch.zeros((n_envs, robot_1k.n_dofs), dtype=gs.tc_float, device=gs.device)

    for step in range(n_steps):
        print("step", step)
        forces.uniform_(-max_force, max_force)
        robot_1k.control_dofs_force(forces)
        robot_2k.control_dofs_force(forces)

        scene_1k.step()
        scene_2k.step()

        # --- broadphase pair count must match per environment ---
        n_pairs_1k, sorted_1k = _get_sorted_broad_pairs(scene_1k)
        n_pairs_2k, sorted_2k = _get_sorted_broad_pairs(scene_2k)

        np.testing.assert_array_equal(
            n_pairs_1k,
            n_pairs_2k,
            err_msg=f"Step {step}: per-env broadphase pair count mismatch",
        )

        if sorted_1k is None:
            print("no sorted 1k")
            continue

        # --- sorted collision pairs must be identical ---
        n_sum = 0
        non_zero_n_env_count = 0
        for i_b in range(n_envs):
            n = n_pairs_1k[i_b]
            if n == 0:
                continue
            n_sum += n
            non_zero_n_env_count += 1
            np.testing.assert_array_equal(
                sorted_1k[i_b, :n],
                sorted_2k[i_b, :n],
                err_msg=f"Step {step}, env {i_b}: broadphase pairs mismatch",
            )
        print("n_sum", n_sum, "non_zero_n_env_count", non_zero_n_env_count)
