"""Verify H patching correctness by comparing patched H against reference full rebuild."""
import sys
import types
import numpy as np

sys.path.insert(0, "tests")
tests_pkg = types.ModuleType("tests")
tests_pkg.__path__ = ["tests"]
sys.modules["tests"] = tests_pkg

import genesis as gs
gs.init()

from tests.test_rigid_benchmarks import make_dex_hand
import quadrants as qd

N_ENVS = 4
WARMUP = 5

scene, step_fn, _ = make_dex_hand(N_ENVS)
qd.sync()

for i in range(WARMUP):
    step_fn()
qd.sync()

csolver = scene._sim.rigid_solver.constraint_solver
cs = csolver.constraint_state
rigid_info = scene._sim.rigid_solver._rigid_global_info

n_dofs = cs.nt_H.shape[1]
B = cs.nt_H.shape[0]
print(f"n_dofs={n_dofs}, B={B}")

# Read state
nt_H = cs.nt_H.to_numpy()
jac = cs.jac.to_numpy()
efc_D = cs.efc_D.to_numpy()
active = cs.active.to_numpy()
mass_mat = rigid_info.mass_mat.to_numpy()
n_constraints = cs.n_constraints.to_numpy()

i_b = 0
nc = n_constraints[i_b]

M = mass_mat[:, :, i_b]
J = jac[:nc, :, i_b]
D = efc_D[:nc, i_b]
act = active[:nc, i_b].astype(float)
H_actual = nt_H[i_b]

# Diagnose the bad element: H[58,58]
i, j = 58, 58
print(f"\n=== Diagnosing H[{i},{j}] for env {i_b} ===")
print(f"  M[{i},{j}] = {M[i,j]:.10f}")
print(f"  H_actual[{i},{j}] = {H_actual[i,j]:.10f}")

jd_contrib = 0.0
for c in range(nc):
    if J[c, i] != 0 and D[c] != 0 and act[c] != 0:
        contrib = D[c] * act[c] * J[c, i] * J[c, j]
        jd_contrib += contrib
        print(f"  c={c}: D={D[c]:.6f}, active={act[c]:.0f}, "
              f"J[c,{i}]={J[c,i]:.8f}, J[c,{j}]={J[c,j]:.8f}, contrib={contrib:.8f}")

print(f"  Sum of J^T D J contributions: {jd_contrib:.10f}")
print(f"  M + sum = {M[i,j] + jd_contrib:.10f}")
print(f"  Expected - Actual = {M[i,j] + jd_contrib - H_actual[i,j]:.10f}")

# Check if there's an "improved" flag
improved = cs.improved.to_numpy()
print(f"\n  improved[{i_b}] = {improved[i_b]}")

# Check the full H lower triangle sum
h_actual_sum = 0.0
h_ref_sum = 0.0
D_masked = D * act
H_ref = M.copy()
for c in range(nc):
    if D_masked[c] != 0:
        H_ref += D_masked[c] * np.outer(J[c], J[c])

for ii in range(n_dofs):
    for jj in range(ii + 1):
        h_actual_sum += abs(H_actual[ii, jj])
        h_ref_sum += abs(H_ref[ii, jj])

print(f"\n  L1 norm lower tri: actual={h_actual_sum:.4f}, ref={h_ref_sum:.4f}")

# Check which elements have large errors
print("\n  Elements with error > 1.0:")
for ii in range(n_dofs):
    for jj in range(ii + 1):
        err = abs(float(H_actual[ii, jj]) - float(H_ref[ii, jj]))
        if err > 1.0:
            print(f"    H[{ii},{jj}]: actual={H_actual[ii,jj]:.6f}, ref={H_ref[ii,jj]:.6f}, "
                  f"M={M[ii,jj]:.6f}, err={err:.6f}")

# Maybe the Hessian only covers a subset of DOFs?
# Check which DOFs actually have nonzero mass entries
print(f"\n  Mass matrix diagonal (last 10 DOFs):")
for d in range(max(0, n_dofs-10), n_dofs):
    print(f"    M[{d},{d}] = {M[d,d]:.8f}")

# Check if there's an island structure
print(f"\n  Contact island info:")
print(f"    csolver._use_contact_island: {getattr(scene._sim.rigid_solver, '_use_contact_island', 'N/A')}")
print(f"    type(csolver): {type(csolver).__name__}")
