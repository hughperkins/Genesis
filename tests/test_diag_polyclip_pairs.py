"""Diagnostic: enumerate the geom types in the dex_hand benchmark scene and bucket
the broadphase-eligible collision pairs by (type_a, type_b). Lets us see what
fraction of multi-contact work polyclip currently catches (mesh-mesh only) vs
what extending polyclip to box-mesh / cylinder-mesh would unlock.

Marked @pytest.mark.required so it picks up `unit_tests_cluster.py --cpu`.
Prints to stdout with a [DIAG] tag so cluster log scraping picks it out.
"""

from collections import Counter

import pytest

import genesis as gs


@pytest.mark.required
def test_diag_polyclip_pairs_dex_hand(show_viewer):
    from .test_rigid_benchmarks import make_dex_hand

    scene, _step, _meta = make_dex_hand(n_envs=1)

    geoms_info = scene.rigid_solver.geoms_info
    types = geoms_info.type.to_numpy()
    name_of = {int(v): n for n, v in vars(gs.GEOM_TYPE).items() if isinstance(v, int) and not n.startswith("_")}

    print(f"[DIAG] n_geoms = {len(types)}")
    geom_counter = Counter(name_of.get(int(t), str(int(t))) for t in types)
    for k, v in sorted(geom_counter.items(), key=lambda kv: -kv[1]):
        print(f"[DIAG]   geom {k}: {v}")

    pair_idx = scene.rigid_solver.collider._collision_pair_idx
    pair_counter = Counter()
    n_pairs_total = 0
    for i_ga in range(len(types)):
        for i_gb in range(i_ga + 1, len(types)):
            if int(pair_idx[i_ga, i_gb]) >= 0:
                ta = name_of.get(int(types[i_ga]), str(int(types[i_ga])))
                tb = name_of.get(int(types[i_gb]), str(int(types[i_gb])))
                key = tuple(sorted((ta, tb)))
                pair_counter[key] += 1
                n_pairs_total += 1

    lines = [f"[DIAG] n_geoms = {len(types)}"]
    for k, v in sorted(geom_counter.items(), key=lambda kv: -kv[1]):
        lines.append(f"[DIAG]   geom {k}: {v}")
    lines.append(f"[DIAG] n_collision_pairs_eligible = {n_pairs_total}")
    for k, v in sorted(pair_counter.items(), key=lambda kv: -kv[1]):
        ta, tb = k
        tag = ""
        if ta == "MESH" and tb == "MESH":
            tag = " (polyclip fires today)"
        elif {ta, tb} == {"BOX", "MESH"}:
            tag = " (would fire if polyclip extended to box-mesh)"
        elif {ta, tb} == {"CYLINDER", "MESH"}:
            tag = " (would fire if polyclip extended to cylinder-mesh)"
        lines.append(f"[DIAG]   pair {ta:<10s} x {tb:<10s}: {v}{tag}")

    msg = "\n".join(lines)
    print(msg)
