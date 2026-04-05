# Iterative Linesearch: Old vs New

## 1. Old Iterative Linesearch (`func_linesearch_batch` in `solver.py`)

### Threading

**1 thread per batch element.** Called from `func_solve_body_monolith` which does `for i_b in range(_B)` — each batch element gets its own thread. All loops over DOFs and constraints are plain sequential `for` loops inside that single thread.

### Algorithm — three phases

#### Phase 1: Init

- Sequential loops to compute `mv = M @ search`, `jv = J @ search`, snorm, gtol
- Evaluate p0 at alpha=0 → `p0_cost, p0_grad, p0_hess`
- Newton step: `p1_alpha = -p0_grad / p0_hess`
- Evaluate at p1_alpha → `p1_cost, p1_grad, p1_hess`
- **Key:** If `p0_cost < p1_cost`, **replace p1 with p0** (fall back to alpha=0 if Newton step made things worse)
- If `|p1_grad| < gtol`: done, return p1_alpha (which might be 0 after the fallback)

#### Phase 2: Newton chase (finding a bracket that spans zero)

- Set `direction = sign(p1_grad)`. Copy p1 → p2.
- While `p1_grad` maintains its sign and iterations remain:
  - Save p1 → p2
  - Take a Newton step from p1, evaluate at new alpha
  - If `|p1_grad| < gtol`: done
- **Result:** p2 has the old gradient sign, p1 has flipped to the opposite sign. The bracket `[p1, p2]` now **straddles the gradient zero-crossing**.

#### Phase 3: Bracket refinement

- Three candidate alphas per iteration: Newton from p1, Newton from p2, midpoint(p1, p2)
- Batch-evaluate all 3 in a single constraint loop pass (`func_ls_point_fn_3alphas_opt`)
- **Convergence check:** If any candidate has `|grad| < gtol`, pick the lowest-cost converged one → done
- Otherwise, `update_bracket_no_eval_local` on **both** p1 and p2 independently — each bracket point accepts candidates with same-sign gradient but closer to zero
- If neither bracket point updated: return the midpoint → done
- Final fallback after max iterations: return whichever of p1, p2 improved over p0; else return 0

---

## 2. New Iterative Bracket-Search (`_func_iterative_linesearch` in `solver_breakdown.py`)

### Threading

**32 threads per batch element.** Uses `block_dim=32`, `for i_flat in range(_B * 32)`, with `tid = i_flat % 32`, `i_b = i_flat // 32`. All threads in a block cooperate via shared memory arrays and tree reductions (`_reduce_3`, `_reduce_9`). DOF and constraint loops are strided by `tid`, so work is divided across threads.

### Algorithm — two phases

#### Phase 1: Init (cooperative)

- 32 threads cooperatively compute `mv`, `jv`, snorm, DOF quadratic coefficients — strided loops, shared memory reductions
- Cooperative p0 evaluation: constraint contributions split across threads, reduced
- Newton step: `init_alpha = -p0_grad / p0_hess`
- Cooperative eval at init_alpha (friction + contact only)
- If `|init_grad| < gtol` AND `init_cost < p0_cost`: done, `best_alpha = init_alpha`

#### Phase 2: Bracket iteration (cooperative, no chase phase)

- Setup bracket directly from p0 and init_alpha: `lo` = endpoint with more-negative gradient, `hi` = endpoint with more-positive gradient
- Loop (bounded by `ls_iterations`):
  - Three candidates: Newton from lo, Newton from hi, midpoint(lo, hi)
  - Cooperative 3-alpha constraint eval — 9 shared arrays, strided loop, `_reduce_9`
  - Try each candidate against lo and hi using `_tighter_bracket` (same-sign, closer to zero)
  - Converged if: no swap happened, or bracket endpoint gradient is small enough
  - Track `best_alpha` = bracket endpoint with lower cost, if improved over p0
- Apply `best_alpha` cooperatively (strided DOF/constraint update)

---

## 3. Differences

| Aspect | Old | New |
|---|---|---|
| **Threads** | 1 per batch element | 32 per batch element, cooperative |
| **p0 cost fallback** | If Newton step is worse than p0, **falls back to p0** before anything else | No fallback — goes straight to bracket setup from both points |
| **Newton chase (Phase 2)** | Follows Newton steps until gradient **changes sign**, guaranteeing bracket spans zero | **Missing entirely** — bracket is set up from just p0 and the first Newton step |
| **Candidate convergence check** | Each iteration checks if any of 3 candidates has `\|grad\| < gtol`, picks lowest-cost converged one | Only checks convergence at bracket endpoints via `ls_done` |
| **Bracket representation** | Two independent points (p1, p2), each updated independently | Single `[lo, hi]` pair |
| **No-progress fallback** | Returns midpoint alpha | Returns best bracket endpoint if improved, else 0 |
| **Final exhaustion fallback** | Returns better of p1/p2 if either beat p0, else 0 | `best_alpha` tracked throughout iteration |

### Most impactful difference

The missing Newton chase. The old code guarantees the bracket spans zero before starting refinement. The new code forms its bracket from just two points (p0 and the first Newton step), which may have same-sign gradients when the cost landscape has kinks from piecewise friction/contact constraints. When that happens, the bracket refinement can't converge properly, producing suboptimal alphas that accumulate over hundreds of simulation steps into visible regressions.
