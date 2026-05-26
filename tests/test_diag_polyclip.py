"""Diagnostic test for the MPR polyclip wire-in.

Reproduces ``test_collision_edge_cases`` mode 4 (small mesh box dropped onto big
mesh box at corner) and dumps per-step contact data + free-joint state so we
can diff the polyclip code path against the perturbation code path.

The intent is to compare two checkouts of the same branch:
- ``hp/narrow-poly-clip-mpr-mesh`` (HEAD ``d4857d5a``): polyclip wire-in active.
- ``hp/diag-baseline-perturbation`` (sibling branch with the wire-in reverted):
  perturbation-only baseline.

Run via ``cmp-tooling/unit_tests_cluster.py --backend field
-k test_diag_polyclip``. Output goes to stdout (captured by pytest) and to
``/tmp/diag_polyclip_<branch>.json`` so we can scp it back and diff offline.

Not marked ``@pytest.mark.required`` so it is skipped on full unit-test runs.
"""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import trimesh

import genesis as gs
from genesis.utils.misc import qd_to_numpy


def _build_mode4_xml(asset_tmp_path) -> str:
    """Reproduces ``collision_edge_cases`` fixture mode 4 from
    ``tests/test_rigid_physics.py``: small mesh box dropped from z=0.7 onto
    big mesh box at the corner (-0.758, -0.758)."""
    assets = {}
    for i, box_size in enumerate(((0.8, 0.8, 0.04), (0.04, 0.04, 0.005))):
        tmesh = trimesh.creation.box(extents=np.array(box_size) * 2)
        mesh_path = str(asset_tmp_path / f"box{i}.obj")
        tmesh.export(mesh_path, file_type="obj")
        assets[f"box{i}"] = mesh_path

    mjcf = ET.Element("mujoco", model="diag_mode4")
    ET.SubElement(mjcf, "option", timestep="0.005")
    default = ET.SubElement(mjcf, "default")
    ET.SubElement(default, "geom", contype="1", conaffinity="1", condim="3", friction="1. 0.5 0.5")

    asset = ET.SubElement(mjcf, "asset")
    for name, mesh_path in assets.items():
        ET.SubElement(asset, "mesh", name=name, refpos="0 0 0", refquat="1 0 0 0", file=mesh_path)

    worldbody = ET.SubElement(mjcf, "worldbody")
    ET.SubElement(worldbody, "geom", type="mesh", mesh="box0", pos="0. 0. 0.", rgba="0 1 0 0.4")
    box1_body = ET.SubElement(worldbody, "body", name="box1", pos="0.0 0.0 0.7")
    ET.SubElement(box1_body, "geom", type="mesh", mesh="box1", pos="-0.758 -0.758 0.", rgba="0 0 1 0.4")
    ET.SubElement(box1_body, "joint", name="root", type="free")

    out_path = str(asset_tmp_path / "diag_mode4.xml")
    ET.ElementTree(mjcf).write(out_path)
    return out_path


def _dump_step(solver, env_idx: int = 0) -> dict:
    """Read everything we need for one diagnostic step."""
    cs = solver.collider._collider_state
    n = int(qd_to_numpy(cs.n_contacts)[env_idx])
    contacts = []
    if n > 0:
        for i in range(n):
            contacts.append(
                {
                    "geom_a": int(qd_to_numpy(cs.contact_data.geom_a)[i, env_idx]),
                    "geom_b": int(qd_to_numpy(cs.contact_data.geom_b)[i, env_idx]),
                    "pos": qd_to_numpy(cs.contact_data.pos)[i, env_idx].tolist(),
                    "normal": qd_to_numpy(cs.contact_data.normal)[i, env_idx].tolist(),
                    "penetration": float(qd_to_numpy(cs.contact_data.penetration)[i, env_idx]),
                    "force": qd_to_numpy(cs.contact_data.force)[i, env_idx].tolist(),
                }
            )
    return {
        "n_contacts": n,
        "contacts": contacts,
        "qpos": solver.get_qpos().cpu().numpy().tolist(),
        "qvel": solver.get_dofs_velocity().cpu().numpy().tolist(),
    }


def _summarize(steps: list[dict]) -> dict:
    """Aggregate stats useful for diagnosing the rest-state regression."""
    n_steps = len(steps)
    n_contacts = [s["n_contacts"] for s in steps]
    final = steps[-1]

    # Variation in contact normal across contacts at each step (measures whether
    # all contacts share an identical normal vs perturbation-style noise).
    normal_spreads = []
    for s in steps:
        if s["n_contacts"] >= 2:
            ns = np.array([c["normal"] for c in s["contacts"]])
            mean = ns.mean(axis=0)
            mean = mean / (np.linalg.norm(mean) + 1e-30)
            dots = ns @ mean
            spread = float(1.0 - dots.min())  # 0 = identical, larger = more spread
            normal_spreads.append(spread)

    return {
        "n_steps": n_steps,
        "n_contacts_min": int(min(n_contacts)) if n_contacts else 0,
        "n_contacts_max": int(max(n_contacts)) if n_contacts else 0,
        "n_contacts_mean": float(np.mean(n_contacts)) if n_contacts else 0.0,
        "normal_spread_mean": float(np.mean(normal_spreads)) if normal_spreads else 0.0,
        "normal_spread_max": float(np.max(normal_spreads)) if normal_spreads else 0.0,
        "final_qpos": final["qpos"],
        "final_qvel": final["qvel"],
        "final_n_contacts": final["n_contacts"],
        "final_contacts": final["contacts"],
    }


@pytest.mark.parametrize("backend", [gs.cpu])
def test_diag_polyclip_mode4(asset_tmp_path, show_viewer):
    xml_path = _build_mode4_xml(asset_tmp_path)

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(),
        sim_options=gs.options.SimOptions(dt=0.005, substeps=1, gravity=(0.0, 0.0, -9.81)),
        rigid_options=gs.options.RigidOptions(
            integrator=gs.integrator.Euler,
            constraint_solver=gs.constraint_solver.CG,
            enable_mujoco_compatibility=False,
            box_box_detection=True,
            enable_self_collision=True,
            enable_adjacent_collision=True,
            enable_multi_contact=True,
            iterations=100,
            tolerance=1e-8,
            ls_iterations=50,
            ls_tolerance=0.01,
            use_gjk_collision=False,  # MPR CCD - the path my polyclip wire-in lives on
        ),
        show_viewer=show_viewer,
        show_FPS=False,
    )
    scene.add_entity(
        gs.morphs.MJCF(file=xml_path, convexify=True, decompose_robot_error_threshold=float("inf")),
        visualize_contact=False,
    )
    scene.build()

    solver = scene.rigid_solver
    qpos_0 = solver.get_qpos().cpu().numpy()
    print(f"[diag] initial qpos = {qpos_0.tolist()}")

    steps = []
    for k in range(200):
        scene.step()
        steps.append(_dump_step(solver))

    summary = _summarize(steps)

    # Print summary to stdout so unit_tests_cluster.py captures it for offline diff.
    # Use a stable [DIAG] prefix so we can grep it out of pytest output.
    print(f"[DIAG] qpos_0 = {qpos_0.tolist()}")
    print("[DIAG] summary:")
    for k, v in summary.items():
        if k in ("final_contacts",):
            continue
        print(f"[DIAG]   {k} = {v}")

    print("[DIAG] final contacts:")
    for i, c in enumerate(summary["final_contacts"]):
        print(
            f"[DIAG]   contact {i}: pos={c['pos']} normal={c['normal']} "
            f"pen={c['penetration']:.3e} force={c['force']}"
        )

    # Also dump per-step n_contacts and the rotational quaternion components to
    # see settling dynamics. qpos[3:7] are quat (wxyz); we just print qx,qy,qz.
    print("[DIAG] per-step n_contacts and free-joint orientation drift:")
    for k_step, s in enumerate(steps):
        if k_step % 10 == 0 or k_step in (0, 1, 2, 3, 4, 5, 49, 99, 199):
            qpos = s["qpos"]
            quat_xyz = qpos[4:7] if len(qpos) >= 7 else qpos
            qvel = s["qvel"]
            print(
                f"[DIAG]   step {k_step:3d}: n_contacts={s['n_contacts']} "
                f"qx,qy,qz={quat_xyz} qvel_norm={float(np.linalg.norm(qvel)):.3e}"
            )

    # Also write json for offline inspection. Use the host-mounted worktree dir as
    # the default; cluster mounts $HOME so /mnt/data/hugh/... is fine.
    out_dir = os.environ.get("DIAG_OUT_DIR", os.getcwd())
    branch_tag = os.environ.get("DIAG_BRANCH_TAG", "unknown")
    out_path = os.path.join(out_dir, f"diag_polyclip_mode4_{branch_tag}.json")
    try:
        with open(out_path, "w") as f:
            json.dump({"steps": steps, "summary": summary, "qpos_0": qpos_0.tolist()}, f)
        print(f"[DIAG] wrote {out_path}")
    except OSError as e:
        print(f"[DIAG] could not write json: {e}")
