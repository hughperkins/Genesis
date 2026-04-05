import numpy as np
import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.engine.solvers.rigid.constraint import solver

# --- Parallel linesearch constants ---
# Number of candidate step sizes evaluated simultaneously per env.
# Each CUDA block processes one env with K threads, using shared memory for the argmin reduction.
# Similar to BLOCK_DIM in func_hessian_direct_tiled: determines parallelism and shared memory layout.
LS_PARALLEL_K = 32

# Floor for the Newton step estimate used to center the log-spaced search range.
# When |grad/hess| is near-zero the search range [alpha*1e-2, alpha*1e2] would collapse;
# this clamp keeps the range meaningful. The value is well below typical linesearch tolerances
# (ls_tolerance * tolerance ~ 1e-2 * 1e-8 for double, ~ 1e-2 * 1e-5 for float) so it never
# masks a genuinely small optimal step.
LS_PARALLEL_MIN_STEP = 1e-6

# Block sizes for shared-memory reductions in _kernel_parallel_linesearch_p0 and _jv.
_P0_BLOCK = 32
_JV_BLOCK = 32

# Maximum bisection iterations for gradient-guided refinement after grid search.
LS_BISECT_STEPS = 12

# Number of alpha candidates evaluated via cooperative constraint reduction.
# Each candidate is evaluated by ALL K threads cooperating on the constraint sum,
# reducing per-thread work from O(n_constraints) to O(n_constraints/K).
LS_N_CANDIDATES = 6

# Maximum allowed alpha (prevents divergence from degenerate steps).
LS_ALPHA_MAX = 1e4


@qd.func
def _ls_eval_cost_grad(
    alpha,
    i_b,
    constraint_state: array_class.ConstraintState,
):
    """Compute cost and analytical gradient at alpha (thread-0 only).

    Follows the same quadratic-coefficient approach as func_ls_point_fn_opt in solver.py.
    Reuses quad_gauss and eq_sum precomputed by the p0 kernel.
    Returns (cost, grad).
    """
    ne = constraint_state.n_constraints_equality[i_b]
    nef = ne + constraint_state.n_constraints_frictionloss[i_b]
    n_con = constraint_state.n_constraints[i_b]

    # Start from precomputed DOF + equality coefficients
    qt_0 = constraint_state.quad_gauss[0, i_b] + constraint_state.eq_sum[0, i_b]
    qt_1 = constraint_state.quad_gauss[1, i_b] + constraint_state.eq_sum[1, i_b]
    qt_2 = constraint_state.quad_gauss[2, i_b] + constraint_state.eq_sum[2, i_b]

    # Friction constraints: accumulate activation-dependent quad coefficients
    for i_c in range(ne, nef):
        Jaref_c = constraint_state.Jaref[i_c, i_b]
        jv_c = constraint_state.jv[i_c, i_b]
        D = constraint_state.efc_D[i_c, i_b]
        f_val = constraint_state.efc_frictionloss[i_c, i_b]
        r_val = constraint_state.diag[i_c, i_b]
        qf_0 = D * (0.5 * Jaref_c * Jaref_c)
        qf_1 = D * (jv_c * Jaref_c)
        qf_2 = D * (0.5 * jv_c * jv_c)
        x = Jaref_c + alpha * jv_c
        rf = r_val * f_val
        linear_neg = x <= -rf
        linear_pos = x >= rf
        if linear_neg or linear_pos:
            qf_0 = linear_neg * f_val * (-0.5 * rf - Jaref_c) + linear_pos * f_val * (-0.5 * rf + Jaref_c)
            qf_1 = linear_neg * (-f_val * jv_c) + linear_pos * (f_val * jv_c)
            qf_2 = 0.0
        qt_0 = qt_0 + qf_0
        qt_1 = qt_1 + qf_1
        qt_2 = qt_2 + qf_2

    # Contact constraints: active when x < 0
    for i_c in range(nef, n_con):
        Jaref_c = constraint_state.Jaref[i_c, i_b]
        jv_c = constraint_state.jv[i_c, i_b]
        D = constraint_state.efc_D[i_c, i_b]
        x = Jaref_c + alpha * jv_c
        active = x < 0
        qf_0 = D * (0.5 * Jaref_c * Jaref_c)
        qf_1 = D * (jv_c * Jaref_c)
        qf_2 = D * (0.5 * jv_c * jv_c)
        qt_0 = qt_0 + qf_0 * active
        qt_1 = qt_1 + qf_1 * active
        qt_2 = qt_2 + qf_2 * active

    cost = alpha * alpha * qt_2 + alpha * qt_1 + qt_0
    grad = 2.0 * alpha * qt_2 + qt_1
    return cost, grad


# Block size for the iterative bracket-search linesearch.
_LS_ITER_BLOCK = 32


@qd.func
def _eval_constraint_at_alpha(
    alpha,
    i_c,
    i_b,
    ne,
    nef,
    constraint_state: array_class.ConstraintState,
):
    """Evaluate a single constraint's cost/grad/hess contribution at a given alpha.

    Returns (cost, grad, hess) for the variable part (friction + contact only).
    Equality constraints are handled via precomputed constant coefficients.
    """
    Jaref_c = constraint_state.Jaref[i_c, i_b]
    jv_c = constraint_state.jv[i_c, i_b]
    D = constraint_state.efc_D[i_c, i_b]
    x = Jaref_c + alpha * jv_c
    jvD = jv_c * D
    jv2D = jv_c * jvD

    ec = gs.qd_float(0.0)
    eg = gs.qd_float(0.0)
    eh = gs.qd_float(0.0)

    if i_c < nef:
        f_val = constraint_state.efc_frictionloss[i_c, i_b]
        r_val = constraint_state.diag[i_c, i_b]
        rf = r_val * f_val
        if x <= -rf:
            ec = f_val * (-0.5 * rf - x)
            eg = -f_val * jv_c
        elif x >= rf:
            ec = f_val * (-0.5 * rf + x)
            eg = f_val * jv_c
        else:
            ec = 0.5 * D * x * x
            eg = jvD * x
            eh = jv2D
    else:
        if x < 0:
            ec = 0.5 * D * x * x
            eg = jvD * x
            eh = jv2D

    return ec, eg, eh


@qd.func
def _reduce_3(sh_a, sh_b, sh_c, tid):
    """Tree-reduce 3 shared arrays of size _LS_ITER_BLOCK in-place. Result in index 0."""
    _K = qd.static(_LS_ITER_BLOCK)
    qd.simt.block.sync()
    stride = _K // 2
    while stride > 0:
        if tid < stride:
            sh_a[tid] += sh_a[tid + stride]
            sh_b[tid] += sh_b[tid + stride]
            sh_c[tid] += sh_c[tid + stride]
        qd.simt.block.sync()
        stride //= 2


@qd.func
def _reduce_6(sh_a, sh_b, sh_c, sh_d, sh_e, sh_f, tid):
    """Tree-reduce 6 shared arrays of size _LS_ITER_BLOCK in-place. Result in index 0."""
    _K = qd.static(_LS_ITER_BLOCK)
    qd.simt.block.sync()
    stride = _K // 2
    while stride > 0:
        if tid < stride:
            sh_a[tid] += sh_a[tid + stride]
            sh_b[tid] += sh_b[tid + stride]
            sh_c[tid] += sh_c[tid + stride]
            sh_d[tid] += sh_d[tid + stride]
            sh_e[tid] += sh_e[tid + stride]
            sh_f[tid] += sh_f[tid + stride]
        qd.simt.block.sync()
        stride //= 2


@qd.func
def _reduce_9(sh_a, sh_b, sh_c, sh_d, sh_e, sh_f, sh_g, sh_h, sh_i, tid):
    """Tree-reduce 9 shared arrays of size _LS_ITER_BLOCK in-place. Result in index 0."""
    _K = qd.static(_LS_ITER_BLOCK)
    qd.simt.block.sync()
    stride = _K // 2
    while stride > 0:
        if tid < stride:
            sh_a[tid] += sh_a[tid + stride]
            sh_b[tid] += sh_b[tid + stride]
            sh_c[tid] += sh_c[tid + stride]
            sh_d[tid] += sh_d[tid + stride]
            sh_e[tid] += sh_e[tid + stride]
            sh_f[tid] += sh_f[tid + stride]
            sh_g[tid] += sh_g[tid + stride]
            sh_h[tid] += sh_h[tid + stride]
            sh_i[tid] += sh_i[tid + stride]
        qd.simt.block.sync()
        stride //= 2


@qd.func
def _tighter_bracket(old_grad, new_grad):
    """True if new_grad is closer to zero from the same sign as old_grad."""
    return (old_grad < new_grad and new_grad < 0.0) or (old_grad > new_grad and new_grad > 0.0)


@qd.func
def _func_iterative_linesearch(
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    """Fused iterative bracket-search linesearch with fully cooperative constraint evaluation.

    One block of K threads per env.  All phases — mv, jv, p0 evaluation, bracket search, and alpha apply — are fused
    into a single grid launch.  The bracket iteration evaluates three candidate step sizes per iteration (Newton from
    lo, Newton from hi, midpoint) with all K threads cooperating on the constraint reduction.  No phase serializes to
    a single thread.

    Algorithm:
      1. Cooperative mv = M @ search, jv = J @ search.
      2. Cooperative p0 evaluation (cost, gradient, curvature at alpha=0).
      3. Newton step from p0 → initial bracket endpoint.
      4. Bracket iteration: evaluate 3 candidate alphas cooperatively, swap bracket endpoints toward the gradient
         zero-crossing.
      5. Pick best alpha, apply to qacc, Ma, Jaref.
    """
    _B = constraint_state.grad.shape[1]
    _K = qd.static(_LS_ITER_BLOCK)

    qd.loop_config(name="iterative_linesearch", block_dim=_K)
    for i_flat in range(_B * _K):
        tid = i_flat % _K
        i_b = i_flat // _K

        sh0 = qd.simt.block.SharedArray((_K,), gs.qd_float)
        sh1 = qd.simt.block.SharedArray((_K,), gs.qd_float)
        sh2 = qd.simt.block.SharedArray((_K,), gs.qd_float)
        sh3 = qd.simt.block.SharedArray((_K,), gs.qd_float)
        sh4 = qd.simt.block.SharedArray((_K,), gs.qd_float)
        sh5 = qd.simt.block.SharedArray((_K,), gs.qd_float)
        sh6 = qd.simt.block.SharedArray((_K,), gs.qd_float)
        sh7 = qd.simt.block.SharedArray((_K,), gs.qd_float)
        sh8 = qd.simt.block.SharedArray((_K,), gs.qd_float)

        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            n_dofs = constraint_state.search.shape[0]
            n_con = constraint_state.n_constraints[i_b]
            ne = constraint_state.n_constraints_equality[i_b]
            nef = ne + constraint_state.n_constraints_frictionloss[i_b]
            EPS = rigid_global_info.EPS[None]

            # ── mv = M @ search (cooperative over DOFs) ──────────────────────────────────────────────────────────────
            i_d1 = tid
            while i_d1 < n_dofs:
                I_d1 = [i_d1, i_b] if qd.static(static_rigid_sim_config.batch_dofs_info) else i_d1
                i_e = dofs_info.entity_idx[I_d1]
                mv_val = gs.qd_float(0.0)
                for i_d2 in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
                    mv_val += rigid_global_info.mass_mat[i_d1, i_d2, i_b] * constraint_state.search[i_d2, i_b]
                constraint_state.mv[i_d1, i_b] = mv_val
                i_d1 += _K

            # ── jv = J @ search (cooperative over constraints) ───────────────────────────────────────────────────────
            i_c = tid
            while i_c < n_con:
                jv_val = gs.qd_float(0.0)
                if qd.static(static_rigid_sim_config.sparse_solve):
                    for i_d_ in range(constraint_state.jac_n_relevant_dofs[i_c, i_b]):
                        i_d = constraint_state.jac_relevant_dofs[i_c, i_d_, i_b]
                        jv_val += constraint_state.jac[i_c, i_d, i_b] * constraint_state.search[i_d, i_b]
                else:
                    for i_d in range(n_dofs):
                        jv_val += constraint_state.jac[i_c, i_d, i_b] * constraint_state.search[i_d, i_b]
                constraint_state.jv[i_c, i_b] = jv_val
                i_c += _K

            qd.simt.block.sync()

            # ── DOF reduction: snorm², quad_gauss coefficients ───────────────────────────────────────────────────────
            loc_snorm_sq = gs.qd_float(0.0)
            loc_qg1 = gs.qd_float(0.0)
            loc_qg2 = gs.qd_float(0.0)
            i_d = tid
            while i_d < n_dofs:
                s = constraint_state.search[i_d, i_b]
                loc_snorm_sq += s * s
                loc_qg1 += s * (constraint_state.Ma[i_d, i_b] - dofs_state.force[i_d, i_b])
                loc_qg2 += 0.5 * s * constraint_state.mv[i_d, i_b]
                i_d += _K

            sh0[tid] = loc_snorm_sq
            sh1[tid] = loc_qg1
            sh2[tid] = loc_qg2
            _reduce_3(sh0, sh1, sh2, tid)

            snorm = qd.sqrt(sh0[0])
            qg_0 = constraint_state.gauss[i_b]
            qg_1 = sh1[0]
            qg_2 = sh2[0]

            if snorm < EPS:
                if tid == 0:
                    constraint_state.improved[i_b] = False
            else:
                # ── Constraint reduction: eq_sum + p0 (cost/grad/hess at alpha=0) ────────────────────────────────────
                loc_eq0 = gs.qd_float(0.0)
                loc_eq1 = gs.qd_float(0.0)
                loc_eq2 = gs.qd_float(0.0)
                loc_p0c = gs.qd_float(0.0)
                loc_p0g = gs.qd_float(0.0)
                loc_p0h = gs.qd_float(0.0)

                i_c = tid
                while i_c < n_con:
                    Jaref_c = constraint_state.Jaref[i_c, i_b]
                    jv_c = constraint_state.jv[i_c, i_b]
                    D = constraint_state.efc_D[i_c, i_b]
                    qf_0 = D * (0.5 * Jaref_c * Jaref_c)
                    qf_1 = D * (jv_c * Jaref_c)
                    qf_2 = D * (0.5 * jv_c * jv_c)

                    if i_c < ne:
                        loc_eq0 += qf_0
                        loc_eq1 += qf_1
                        loc_eq2 += qf_2
                        loc_p0c += qf_0
                        loc_p0g += qf_1
                        loc_p0h += qf_2
                    elif i_c < nef:
                        f = constraint_state.efc_frictionloss[i_c, i_b]
                        r = constraint_state.diag[i_c, i_b]
                        rf = r * f
                        ln = Jaref_c <= -rf
                        lp = Jaref_c >= rf
                        if ln or lp:
                            qf_0 = ln * f * (-0.5 * rf - Jaref_c) + lp * f * (-0.5 * rf + Jaref_c)
                            qf_1 = ln * (-f * jv_c) + lp * (f * jv_c)
                            qf_2 = 0.0
                        loc_p0c += qf_0
                        loc_p0g += qf_1
                        loc_p0h += qf_2
                    else:
                        active = Jaref_c < 0
                        loc_p0c += qf_0 * active
                        loc_p0g += qf_1 * active
                        loc_p0h += qf_2 * active

                    i_c += _K

                sh0[tid] = loc_eq0
                sh1[tid] = loc_eq1
                sh2[tid] = loc_eq2
                sh3[tid] = loc_p0c
                sh4[tid] = loc_p0g
                sh5[tid] = loc_p0h
                _reduce_6(sh0, sh1, sh2, sh3, sh4, sh5, tid)

                # Constant quadratic coefficients (DOF + equality), reused for every alpha evaluation
                const_0 = qg_0 + sh0[0]
                const_1 = qg_1 + sh1[0]
                const_2 = qg_2 + sh2[0]

                p0_cost = qg_0 + sh3[0]
                p0_grad = qg_1 + sh4[0]
                p0_hess = 2.0 * (qg_2 + sh5[0])
                if p0_hess <= 0.0:
                    p0_hess = EPS

                # Adaptive linesearch tolerance
                scale = rigid_global_info.meaninertia[i_b] * qd.max(1, n_dofs)
                gtol = qd.max(
                    rigid_global_info.tolerance[None] * rigid_global_info.ls_tolerance[None] * snorm * scale, EPS
                )

                # ── Initial Newton step from p0 ─────────────────────────────────────────────────────────────────────
                init_alpha = -p0_grad / p0_hess

                # ── Cooperative eval at init_alpha (friction + contact only) ─────────────────────────────────────────
                loc_vc = gs.qd_float(0.0)
                loc_vg = gs.qd_float(0.0)
                loc_vh = gs.qd_float(0.0)
                i_c = ne + tid
                while i_c < n_con:
                    ec, eg, eh = _eval_constraint_at_alpha(
                        init_alpha, i_c, i_b, ne, nef, constraint_state
                    )
                    loc_vc += ec
                    loc_vg += eg
                    loc_vh += eh
                    i_c += _K

                sh0[tid] = loc_vc
                sh1[tid] = loc_vg
                sh2[tid] = loc_vh
                _reduce_3(sh0, sh1, sh2, tid)

                init_cost = const_0 + init_alpha * const_1 + init_alpha * init_alpha * const_2 + sh0[0]
                init_grad = const_1 + 2.0 * init_alpha * const_2 + sh1[0]
                init_hess = 2.0 * const_2 + sh2[0]
                if init_hess <= 0.0:
                    init_hess = EPS

                # ── Phase 1b: p0 cost fallback (match old linesearch) ──────────────────────────────────────────────
                best_alpha = gs.qd_float(0.0)
                max_ls_iter = rigid_global_info.ls_iterations[None]

                cur_a = init_alpha
                cur_c = init_cost
                cur_g = init_grad
                cur_h = init_hess
                if p0_cost < init_cost:
                    cur_a = gs.qd_float(0.0)
                    cur_c = p0_cost
                    cur_g = p0_grad
                    cur_h = p0_hess

                if qd.abs(cur_g) < gtol:
                    best_alpha = cur_a
                else:
                    # ── Phase 2: Newton chase — follow Newton steps until gradient sign change ──────────────────
                    direction = gs.qd_int(1) if cur_g < 0.0 else gs.qd_int(-1)
                    prev_a = cur_a
                    prev_c = cur_c
                    prev_g = cur_g
                    prev_h = cur_h
                    prev_updated = False
                    chase_done = False
                    ls_iter = 0

                    while cur_g * direction <= -gtol and ls_iter < max_ls_iter:
                        ls_iter += 1
                        prev_a = cur_a
                        prev_c = cur_c
                        prev_g = cur_g
                        prev_h = cur_h
                        prev_updated = True

                        next_a = cur_a - cur_g / cur_h

                        # ── Cooperative eval at next_a (friction + contact) ────────────────────────────────────
                        loc_vc = gs.qd_float(0.0)
                        loc_vg = gs.qd_float(0.0)
                        loc_vh = gs.qd_float(0.0)
                        i_c = ne + tid
                        while i_c < n_con:
                            ec, eg, eh = _eval_constraint_at_alpha(
                                next_a, i_c, i_b, ne, nef, constraint_state
                            )
                            loc_vc += ec
                            loc_vg += eg
                            loc_vh += eh
                            i_c += _K

                        sh0[tid] = loc_vc
                        sh1[tid] = loc_vg
                        sh2[tid] = loc_vh
                        _reduce_3(sh0, sh1, sh2, tid)

                        cur_a = next_a
                        cur_c = const_0 + next_a * const_1 + next_a * next_a * const_2 + sh0[0]
                        cur_g = const_1 + 2.0 * next_a * const_2 + sh1[0]
                        cur_h = 2.0 * const_2 + sh2[0]
                        if cur_h <= 0.0:
                            cur_h = EPS

                        if qd.abs(cur_g) < gtol:
                            best_alpha = cur_a
                            chase_done = True

                    if not chase_done:
                        if ls_iter >= max_ls_iter:
                            if cur_c < p0_cost:
                                best_alpha = cur_a
                        elif not prev_updated:
                            if cur_c < p0_cost:
                                best_alpha = cur_a
                        else:
                            # ── Phase 3: Bracket refinement ────────────────────────────────────────────────────
                            # prev and cur now bracket the gradient zero-crossing
                            lo_a = gs.qd_float(0.0)
                            lo_c = gs.qd_float(0.0)
                            lo_g = gs.qd_float(0.0)
                            lo_h = gs.qd_float(0.0)
                            hi_a = gs.qd_float(0.0)
                            hi_c = gs.qd_float(0.0)
                            hi_g = gs.qd_float(0.0)
                            hi_h = gs.qd_float(0.0)
                            if cur_g < prev_g:
                                lo_a = cur_a
                                lo_c = cur_c
                                lo_g = cur_g
                                lo_h = cur_h
                                hi_a = prev_a
                                hi_c = prev_c
                                hi_g = prev_g
                                hi_h = prev_h
                            else:
                                lo_a = prev_a
                                lo_c = prev_c
                                lo_g = prev_g
                                lo_h = prev_h
                                hi_a = cur_a
                                hi_c = cur_c
                                hi_g = cur_g
                                hi_h = cur_h

                            ls_done = False
                            while not ls_done and ls_iter < max_ls_iter:
                                ls_iter += 1

                                cand_a = lo_a - lo_g / lo_h
                                cand_b = hi_a - hi_g / hi_h
                                cand_c = 0.5 * (lo_a + hi_a)

                                # ── Cooperative 3-alpha constraint eval (friction + contact) ───────────────────
                                loc_ac = gs.qd_float(0.0)
                                loc_ag = gs.qd_float(0.0)
                                loc_ah = gs.qd_float(0.0)
                                loc_bc = gs.qd_float(0.0)
                                loc_bg = gs.qd_float(0.0)
                                loc_bh = gs.qd_float(0.0)
                                loc_cc = gs.qd_float(0.0)
                                loc_cg = gs.qd_float(0.0)
                                loc_ch = gs.qd_float(0.0)

                                i_c = ne + tid
                                while i_c < n_con:
                                    Jaref_c = constraint_state.Jaref[i_c, i_b]
                                    jv_c = constraint_state.jv[i_c, i_b]
                                    D = constraint_state.efc_D[i_c, i_b]
                                    jvD = jv_c * D
                                    jv2D = jv_c * jvD

                                    xa = Jaref_c + cand_a * jv_c
                                    xb = Jaref_c + cand_b * jv_c
                                    xc = Jaref_c + cand_c * jv_c

                                    if i_c < nef:
                                        f_val = constraint_state.efc_frictionloss[i_c, i_b]
                                        r_val = constraint_state.diag[i_c, i_b]
                                        rf = r_val * f_val

                                        if xa <= -rf:
                                            loc_ac += f_val * (-0.5 * rf - xa)
                                            loc_ag += -f_val * jv_c
                                        elif xa >= rf:
                                            loc_ac += f_val * (-0.5 * rf + xa)
                                            loc_ag += f_val * jv_c
                                        else:
                                            loc_ac += 0.5 * D * xa * xa
                                            loc_ag += jvD * xa
                                            loc_ah += jv2D

                                        if xb <= -rf:
                                            loc_bc += f_val * (-0.5 * rf - xb)
                                            loc_bg += -f_val * jv_c
                                        elif xb >= rf:
                                            loc_bc += f_val * (-0.5 * rf + xb)
                                            loc_bg += f_val * jv_c
                                        else:
                                            loc_bc += 0.5 * D * xb * xb
                                            loc_bg += jvD * xb
                                            loc_bh += jv2D

                                        if xc <= -rf:
                                            loc_cc += f_val * (-0.5 * rf - xc)
                                            loc_cg += -f_val * jv_c
                                        elif xc >= rf:
                                            loc_cc += f_val * (-0.5 * rf + xc)
                                            loc_cg += f_val * jv_c
                                        else:
                                            loc_cc += 0.5 * D * xc * xc
                                            loc_cg += jvD * xc
                                            loc_ch += jv2D
                                    else:
                                        if xa < 0:
                                            loc_ac += 0.5 * D * xa * xa
                                            loc_ag += jvD * xa
                                            loc_ah += jv2D
                                        if xb < 0:
                                            loc_bc += 0.5 * D * xb * xb
                                            loc_bg += jvD * xb
                                            loc_bh += jv2D
                                        if xc < 0:
                                            loc_cc += 0.5 * D * xc * xc
                                            loc_cg += jvD * xc
                                            loc_ch += jv2D

                                    i_c += _K

                                sh0[tid] = loc_ac
                                sh1[tid] = loc_ag
                                sh2[tid] = loc_ah
                                sh3[tid] = loc_bc
                                sh4[tid] = loc_bg
                                sh5[tid] = loc_bh
                                sh6[tid] = loc_cc
                                sh7[tid] = loc_cg
                                sh8[tid] = loc_ch
                                _reduce_9(sh0, sh1, sh2, sh3, sh4, sh5, sh6, sh7, sh8, tid)

                                a_cost = const_0 + cand_a * const_1 + cand_a * cand_a * const_2 + sh0[0]
                                a_grad = const_1 + 2.0 * cand_a * const_2 + sh1[0]
                                a_hess = 2.0 * const_2 + sh2[0]
                                if a_hess <= 0.0:
                                    a_hess = EPS

                                b_cost = const_0 + cand_b * const_1 + cand_b * cand_b * const_2 + sh3[0]
                                b_grad = const_1 + 2.0 * cand_b * const_2 + sh4[0]
                                b_hess = 2.0 * const_2 + sh5[0]
                                if b_hess <= 0.0:
                                    b_hess = EPS

                                c_cost = const_0 + cand_c * const_1 + cand_c * cand_c * const_2 + sh6[0]
                                c_grad = const_1 + 2.0 * cand_c * const_2 + sh7[0]
                                c_hess = 2.0 * const_2 + sh8[0]
                                if c_hess <= 0.0:
                                    c_hess = EPS

                                # ── Convergence check among candidates ─────────────────────────────────────────
                                alphas_0 = cand_a
                                alphas_1 = cand_b
                                alphas_2 = cand_c
                                costs_0 = a_cost
                                costs_1 = b_cost
                                costs_2 = c_cost
                                grads_0 = a_grad
                                grads_1 = b_grad
                                grads_2 = c_grad

                                best_found = False
                                best_cand_cost = gs.qd_float(0.0)
                                if qd.abs(grads_0) < gtol and (not best_found or costs_0 < best_cand_cost):
                                    best_alpha = alphas_0
                                    best_cand_cost = costs_0
                                    best_found = True
                                if qd.abs(grads_1) < gtol and (not best_found or costs_1 < best_cand_cost):
                                    best_alpha = alphas_1
                                    best_cand_cost = costs_1
                                    best_found = True
                                if qd.abs(grads_2) < gtol and (not best_found or costs_2 < best_cand_cost):
                                    best_alpha = alphas_2
                                    best_cand_cost = costs_2
                                    best_found = True

                                if best_found:
                                    ls_done = True
                                else:
                                    # ── Bracket swap ───────────────────────────────────────────────────────────
                                    swap_lo = False
                                    if _tighter_bracket(lo_g, a_grad):
                                        lo_a = cand_a
                                        lo_c = a_cost
                                        lo_g = a_grad
                                        lo_h = a_hess
                                        swap_lo = True
                                    if _tighter_bracket(lo_g, c_grad):
                                        lo_a = cand_c
                                        lo_c = c_cost
                                        lo_g = c_grad
                                        lo_h = c_hess
                                        swap_lo = True
                                    if _tighter_bracket(lo_g, b_grad):
                                        lo_a = cand_b
                                        lo_c = b_cost
                                        lo_g = b_grad
                                        lo_h = b_hess
                                        swap_lo = True

                                    swap_hi = False
                                    if _tighter_bracket(hi_g, b_grad):
                                        hi_a = cand_b
                                        hi_c = b_cost
                                        hi_g = b_grad
                                        hi_h = b_hess
                                        swap_hi = True
                                    if _tighter_bracket(hi_g, c_grad):
                                        hi_a = cand_c
                                        hi_c = c_cost
                                        hi_g = c_grad
                                        hi_h = c_hess
                                        swap_hi = True
                                    if _tighter_bracket(hi_g, a_grad):
                                        hi_a = cand_a
                                        hi_c = a_cost
                                        hi_g = a_grad
                                        hi_h = a_hess
                                        swap_hi = True

                                    if not swap_lo and not swap_hi:
                                        best_alpha = cand_c
                                        ls_done = True
                                    elif (lo_g < 0.0 and lo_g > -gtol) or (hi_g > 0.0 and hi_g < gtol):
                                        if lo_c < p0_cost or hi_c < p0_cost:
                                            if lo_c <= hi_c:
                                                best_alpha = lo_a
                                            else:
                                                best_alpha = hi_a
                                        ls_done = True

                            if not ls_done:
                                if lo_c <= hi_c and lo_c < p0_cost:
                                    best_alpha = lo_a
                                elif hi_c <= lo_c and hi_c < p0_cost:
                                    best_alpha = hi_a

                # ── Apply alpha ──────────────────────────────────────────────────────────────────────────────────────
                if qd.abs(best_alpha) < EPS:
                    if tid == 0:
                        constraint_state.improved[i_b] = False
                else:
                    i_d = tid
                    while i_d < n_dofs:
                        constraint_state.qacc[i_d, i_b] += constraint_state.search[i_d, i_b] * best_alpha
                        constraint_state.Ma[i_d, i_b] += constraint_state.mv[i_d, i_b] * best_alpha
                        i_d += _K
                    i_c = tid
                    while i_c < n_con:
                        constraint_state.Jaref[i_c, i_b] += constraint_state.jv[i_c, i_b] * best_alpha
                        i_c += _K


@qd.func
def _func_parallel_linesearch_p0(
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    """Parallel linesearch P0 kernel: fused mv + jv + snorm + quad_gauss + eq_sum + p0_cost.

    Parallel grid-search linesearch algorithm overview
    --------------------------------------------------
    A block of K=32 threads cooperates on each env. Both approaches are O(n_constraints) per
    evaluation, but the grid search parallelizes each evaluation across 32 threads
    (n_constraints/32 work per thread), whereas the iterative approach runs each evaluation on
    a single thread.

    The algorithm is split across two kernels:

    P0 kernel (this function):
        Phase 0a: Compute mv = M @ search (cooperative over DOFs, 32 threads).
        Phase 0b: Compute jv = J @ search (cooperative over constraints, 32 threads).
        Phase 1: Fused snorm + quad_gauss parallel reduction over n_dofs.
        Phase 2: Parallel reduction over n_constraints for eq_sum and p0_cost.

    Eval kernel (_kernel_parallel_linesearch_eval):
        a) Grid search: Evaluate N_CANDIDATES=6 log-spaced alphas plus the Newton step,
           all 32 threads cooperating on each candidate's constraint reduction.
        b) Newton correction: One Newton step from the best grid candidate. Accepted if it
           improves cost.
        c) Bisection fallback: If Newton fails, bracket the zero-crossing of the gradient
           and bisect up to LS_BISECT_STEPS=12 times.
        d) Apply: Update qacc, Ma, Jaref with the chosen alpha (cooperative over DOFs).

    Post-linesearch: Separate kernels for constraint force update, cost update, gradient
    update, Hessian update (Newton only), and search direction update. These reuse the
    batch-level functions from solver.py.
    """
    _B = constraint_state.grad.shape[1]
    _T = qd.static(_P0_BLOCK)

    qd.loop_config(name="parallel_linesearch_p0", block_dim=_T)
    for i_flat in range(_B * _T):
        tid = i_flat % _T
        i_b = i_flat // _T

        # 6 shared arrays for parallel reductions (reused across phases)
        sh_snorm_sq = qd.simt.block.SharedArray((_T,), gs.qd_float)
        sh_qg_grad = qd.simt.block.SharedArray((_T,), gs.qd_float)
        sh_qg_hess = qd.simt.block.SharedArray((_T,), gs.qd_float)
        sh_p0_cost = qd.simt.block.SharedArray((_T,), gs.qd_float)
        sh_constraint_grad = qd.simt.block.SharedArray((_T,), gs.qd_float)
        sh_constraint_hess = qd.simt.block.SharedArray((_T,), gs.qd_float)

        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            n_dofs = constraint_state.search.shape[0]
            n_con = constraint_state.n_constraints[i_b]

            # === Phase 0a: Compute mv = M @ search (cooperative over DOFs) ===
            i_d1 = tid
            while i_d1 < n_dofs:
                I_d1 = [i_d1, i_b] if qd.static(static_rigid_sim_config.batch_dofs_info) else i_d1
                i_e = dofs_info.entity_idx[I_d1]
                mv_val = gs.qd_float(0.0)
                for i_d2 in range(entities_info.dof_start[i_e], entities_info.dof_end[i_e]):
                    mv_val = mv_val + rigid_global_info.mass_mat[i_d1, i_d2, i_b] * constraint_state.search[i_d2, i_b]
                constraint_state.mv[i_d1, i_b] = mv_val
                i_d1 += _T

            # === Phase 0b: Compute jv = J @ search (cooperative over constraints) ===
            i_c = tid
            while i_c < n_con:
                jv_val = gs.qd_float(0.0)
                if qd.static(static_rigid_sim_config.sparse_solve):
                    for i_d_ in range(constraint_state.jac_n_relevant_dofs[i_c, i_b]):
                        i_d = constraint_state.jac_relevant_dofs[i_c, i_d_, i_b]
                        jv_val = jv_val + constraint_state.jac[i_c, i_d, i_b] * constraint_state.search[i_d, i_b]
                else:
                    for i_d in range(n_dofs):
                        jv_val = jv_val + constraint_state.jac[i_c, i_d, i_b] * constraint_state.search[i_d, i_b]
                constraint_state.jv[i_c, i_b] = jv_val
                i_c += _T

            qd.simt.block.sync()  # Ensure mv and jv are written before Phase 1 reads them

            # === Phase 1: Fused snorm + quad_gauss, parallel over n_dofs ===
            local_snorm_sq = gs.qd_float(0.0)
            local_qg_grad = gs.qd_float(0.0)
            local_qg_hess = gs.qd_float(0.0)

            i_d = tid
            while i_d < n_dofs:
                s = constraint_state.search[i_d, i_b]
                local_snorm_sq += s * s
                local_qg_grad += s * constraint_state.Ma[i_d, i_b] - s * dofs_state.force[i_d, i_b]
                local_qg_hess += 0.5 * s * constraint_state.mv[i_d, i_b]
                i_d += _T

            sh_snorm_sq[tid] = local_snorm_sq
            sh_qg_grad[tid] = local_qg_grad
            sh_qg_hess[tid] = local_qg_hess

            qd.simt.block.sync()

            # Tree reduction for 3 accumulators
            stride = _T // 2
            while stride > 0:
                if tid < stride:
                    sh_snorm_sq[tid] += sh_snorm_sq[tid + stride]
                    sh_qg_grad[tid] += sh_qg_grad[tid + stride]
                    sh_qg_hess[tid] += sh_qg_hess[tid + stride]
                qd.simt.block.sync()
                stride //= 2

            # All threads read the reduced snorm
            snorm = qd.sqrt(sh_snorm_sq[0])

            if snorm < rigid_global_info.EPS[None]:
                # Converged — only thread 0 writes
                if tid == 0:
                    constraint_state.candidates[0, i_b] = 0.0
                    constraint_state.candidates[1, i_b] = 0.0
                    constraint_state.improved[i_b] = False
            else:
                # Thread 0 writes quad_gauss to global memory
                if tid == 0:
                    constraint_state.quad_gauss[0, i_b] = constraint_state.gauss[i_b]
                    constraint_state.quad_gauss[1, i_b] = sh_qg_grad[0]
                    constraint_state.quad_gauss[2, i_b] = sh_qg_hess[0]

                # === Phase 2: Constraint cost, parallel over n_constraints ===
                ne = constraint_state.n_constraints_equality[i_b]
                nef = ne + constraint_state.n_constraints_frictionloss[i_b]
                n_con = constraint_state.n_constraints[i_b]

                local_eq_cost = gs.qd_float(0.0)
                local_eq_grad = gs.qd_float(0.0)
                local_eq_hess = gs.qd_float(0.0)
                local_p0_cost = gs.qd_float(0.0)
                local_constraint_grad = gs.qd_float(0.0)
                local_constraint_hess = gs.qd_float(0.0)

                i_c = tid
                while i_c < n_con:
                    Jaref_c = constraint_state.Jaref[i_c, i_b]
                    jv_c = constraint_state.jv[i_c, i_b]
                    D = constraint_state.efc_D[i_c, i_b]
                    qf_0 = D * (0.5 * Jaref_c * Jaref_c)
                    qf_1 = D * (jv_c * Jaref_c)
                    qf_2 = D * (0.5 * jv_c * jv_c)

                    if i_c < ne:
                        # Equality: always active
                        local_eq_cost += qf_0
                        local_eq_grad += qf_1
                        local_eq_hess += qf_2
                        local_p0_cost += qf_0
                        local_constraint_grad += qf_1
                        local_constraint_hess += qf_2
                    elif i_c < nef:
                        # Friction: check linear regime at alpha=0
                        f = constraint_state.efc_frictionloss[i_c, i_b]
                        r = constraint_state.diag[i_c, i_b]
                        rf = r * f
                        linear_neg = Jaref_c <= -rf
                        linear_pos = Jaref_c >= rf
                        if linear_neg or linear_pos:
                            qf_0 = linear_neg * f * (-0.5 * rf - Jaref_c) + linear_pos * f * (-0.5 * rf + Jaref_c)
                            qf_1 = linear_neg * (-f * jv_c) + linear_pos * (f * jv_c)
                            qf_2 = 0.0
                        local_p0_cost += qf_0
                        local_constraint_grad += qf_1
                        local_constraint_hess += qf_2
                    else:
                        # Contact: active if Jaref < 0
                        active = Jaref_c < 0
                        local_p0_cost += qf_0 * active
                        local_constraint_grad += qf_1 * active
                        local_constraint_hess += qf_2 * active

                    i_c += _T

                # Reuse shared arrays for Phase 2 reduction
                sh_snorm_sq[tid] = local_eq_cost
                sh_qg_grad[tid] = local_eq_grad
                sh_qg_hess[tid] = local_eq_hess
                sh_p0_cost[tid] = local_p0_cost
                sh_constraint_grad[tid] = local_constraint_grad
                sh_constraint_hess[tid] = local_constraint_hess

                qd.simt.block.sync()

                # Tree reduction for 6 accumulators
                stride = _T // 2
                while stride > 0:
                    if tid < stride:
                        sh_snorm_sq[tid] += sh_snorm_sq[tid + stride]
                        sh_qg_grad[tid] += sh_qg_grad[tid + stride]
                        sh_qg_hess[tid] += sh_qg_hess[tid + stride]
                        sh_p0_cost[tid] += sh_p0_cost[tid + stride]
                        sh_constraint_grad[tid] += sh_constraint_grad[tid + stride]
                        sh_constraint_hess[tid] += sh_constraint_hess[tid + stride]
                    qd.simt.block.sync()
                    stride //= 2

                if tid == 0:
                    constraint_state.eq_sum[0, i_b] = sh_snorm_sq[0]
                    constraint_state.eq_sum[1, i_b] = sh_qg_grad[0]
                    constraint_state.eq_sum[2, i_b] = sh_qg_hess[0]
                    constraint_state.ls_it[i_b] = 1
                    constraint_state.candidates[1, i_b] = constraint_state.gauss[i_b] + sh_p0_cost[0]
                    # Initialize best alpha, search range, and best-cost tracker for parallel linesearch
                    constraint_state.candidates[0, i_b] = 0.0  # default: no step

                    # Use full Newton step (DOF + all constraints) as the range center.
                    total_hess = 2.0 * (constraint_state.quad_gauss[2, i_b] + sh_constraint_hess[0])
                    if total_hess > 0.0:
                        total_grad = constraint_state.quad_gauss[1, i_b] + sh_constraint_grad[0]
                        alpha_newton = qd.max(
                            qd.abs(total_grad / total_hess), gs.qd_float(qd.static(LS_PARALLEL_MIN_STEP))
                        )
                        constraint_state.candidates[2, i_b] = alpha_newton * 1e-2
                        constraint_state.candidates[3, i_b] = alpha_newton * 10.0
                        constraint_state.candidates[5, i_b] = alpha_newton  # exact Newton step for eval
                    else:
                        constraint_state.candidates[2, i_b] = 1e-6
                        constraint_state.candidates[3, i_b] = 1e2
                        constraint_state.candidates[5, i_b] = 0.0
                    constraint_state.candidates[4, i_b] = gs.qd_float(1e30)  # best cost across passes
                    # Store gtol for gradient-guided bisection after grid search
                    n_dofs_val = constraint_state.search.shape[0]
                    scale = rigid_global_info.meaninertia[i_b] * qd.max(1, n_dofs_val)
                    constraint_state.candidates[7, i_b] = (
                        rigid_global_info.tolerance[None] * rigid_global_info.ls_tolerance[None] * snorm * scale
                    )


@qd.func
def _func_parallel_linesearch_eval(
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    """Evaluate alpha candidates via cooperative constraint reduction, then bisect.

    All K threads cooperate on each candidate: each thread reduces n_constraints/K
    constraints, then a shared-memory tree reduction sums the partial costs. This is
    O(n_candidates × n_constraints/K) per thread instead of O(K × n_constraints).

    Phase 1: Cooperatively evaluate N_CANDIDATES + Newton alpha, pick best via argmin.
    Phase 2: Cooperatively evaluate analytical gradient at best, try one Newton correction first then bisect if needed.
    """
    _B = constraint_state.grad.shape[1]
    _K = qd.static(LS_PARALLEL_K)
    _NC = qd.static(LS_N_CANDIDATES)

    qd.loop_config(name="parallel_linesearch_eval", block_dim=_K)
    for i_flat in range(_B * _K):
        tid = i_flat % _K
        i_b = i_flat // _K

        # Shared memory for reductions (reused across phases)
        sh_val = qd.simt.block.SharedArray((_K,), gs.qd_float)
        sh_val2 = qd.simt.block.SharedArray((_K,), gs.qd_float)
        # Shared arrays for candidate costs and alphas (only _NC+1 used)
        sh_cand_cost = qd.simt.block.SharedArray((_K,), gs.qd_float)
        sh_cand_alpha = qd.simt.block.SharedArray((_K,), gs.qd_float)

        active = constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]

        if active:
            ne = constraint_state.n_constraints_equality[i_b]
            nef = ne + constraint_state.n_constraints_frictionloss[i_b]
            n_con = constraint_state.n_constraints[i_b]
            lo = constraint_state.candidates[2, i_b]
            hi = constraint_state.candidates[3, i_b]
            p0_cost = constraint_state.candidates[1, i_b]
            gtol = constraint_state.candidates[7, i_b]

            # Pre-compute log-space step for candidate generation
            _log_lo = qd.log(lo)
            _cand_step = (qd.log(hi) - _log_lo) / qd.max(1.0, qd.cast(_NC - 1, gs.qd_float))
            alpha_newton = constraint_state.candidates[5, i_b]

            # === Phase 1: Cooperative evaluation of N_CANDIDATES alphas ===
            # Evaluate each candidate sequentially; all K threads cooperate on constraint reduction.
            n_total_cands = _NC + 1  # +1 for Newton alpha
            for cand_idx in range(n_total_cands):
                # Generate alpha for this candidate
                alpha_c = gs.qd_float(0.0)
                if cand_idx < _NC:
                    alpha_c = qd.exp(_log_lo + qd.cast(cand_idx, gs.qd_float) * _cand_step)
                else:
                    alpha_c = alpha_newton  # last candidate is Newton alpha

                # DOF + equality cost (O(1), same for all threads)
                dof_eq_cost = (
                    alpha_c * alpha_c * constraint_state.quad_gauss[2, i_b]
                    + alpha_c * constraint_state.quad_gauss[1, i_b]
                    + constraint_state.quad_gauss[0, i_b]
                    + alpha_c * alpha_c * constraint_state.eq_sum[2, i_b]
                    + alpha_c * constraint_state.eq_sum[1, i_b]
                    + constraint_state.eq_sum[0, i_b]
                )

                # Cooperative constraint cost: each thread handles strided constraints
                local_cost = gs.qd_float(0.0)
                i_c = ne + tid  # start from ne (skip equality, already in eq_sum)
                while i_c < n_con:
                    Jaref_c = constraint_state.Jaref[i_c, i_b]
                    jv_c = constraint_state.jv[i_c, i_b]
                    D = constraint_state.efc_D[i_c, i_b]
                    x = Jaref_c + alpha_c * jv_c
                    if i_c < nef:
                        # Friction constraint
                        f_val = constraint_state.efc_frictionloss[i_c, i_b]
                        r_val = constraint_state.diag[i_c, i_b]
                        rf = r_val * f_val
                        linear_neg = x <= -rf
                        linear_pos = x >= rf
                        if linear_neg or linear_pos:
                            local_cost = local_cost + linear_neg * f_val * (-0.5 * rf - Jaref_c - alpha_c * jv_c)
                            local_cost = local_cost + linear_pos * f_val * (-0.5 * rf + Jaref_c + alpha_c * jv_c)
                        else:
                            local_cost = local_cost + D * 0.5 * x * x
                    else:
                        # Contact constraint (active if x < 0)
                        if x < 0:
                            local_cost = local_cost + D * 0.5 * x * x
                    i_c += _K

                # Tree reduction for constraint cost
                sh_val[tid] = local_cost
                qd.simt.block.sync()
                stride = _K // 2
                while stride > 0:
                    if tid < stride:
                        sh_val[tid] += sh_val[tid + stride]
                    qd.simt.block.sync()
                    stride //= 2

                # Thread 0 stores total cost for this candidate
                if tid == 0:
                    total_cost = dof_eq_cost + sh_val[0]
                    sh_cand_cost[cand_idx] = total_cost
                    sh_cand_alpha[cand_idx] = alpha_c
                qd.simt.block.sync()

            # === Phase 2: Find best candidate (thread 0) ===
            if tid == 0:
                best_alpha = gs.qd_float(0.0)
                best_cost = p0_cost
                best_cost_prev = constraint_state.candidates[4, i_b]
                for ci in range(n_total_cands):
                    c = sh_cand_cost[ci]
                    if c < best_cost and c < best_cost_prev:
                        best_cost = c
                        best_alpha = sh_cand_alpha[ci]

                constraint_state.candidates[0, i_b] = best_alpha
                if best_alpha > 0.0:
                    constraint_state.candidates[4, i_b] = best_cost
                # Store best alpha for Phase 3 cooperative bisection
                sh_cand_alpha[0] = best_alpha
            qd.simt.block.sync()

            # === Phase 3: Cooperative gradient bisection ===
            best_alpha_shared = sh_cand_alpha[0]
            if best_alpha_shared > 0.0:
                # Cooperatively compute gradient at best_alpha
                alpha_eval = best_alpha_shared

                # Cooperative gradient: accumulate quad_total_1 and quad_total_2
                local_qt1 = gs.qd_float(0.0)
                local_qt2 = gs.qd_float(0.0)
                i_c = ne + tid
                while i_c < n_con:
                    Jaref_c = constraint_state.Jaref[i_c, i_b]
                    jv_c = constraint_state.jv[i_c, i_b]
                    D = constraint_state.efc_D[i_c, i_b]
                    x = Jaref_c + alpha_eval * jv_c
                    if i_c < nef:
                        f_val = constraint_state.efc_frictionloss[i_c, i_b]
                        r_val = constraint_state.diag[i_c, i_b]
                        rf = r_val * f_val
                        linear_neg = x <= -rf
                        linear_pos = x >= rf
                        qf_1 = D * (jv_c * Jaref_c)
                        qf_2 = D * (0.5 * jv_c * jv_c)
                        if linear_neg or linear_pos:
                            qf_1 = linear_neg * (-f_val * jv_c) + linear_pos * (f_val * jv_c)
                            qf_2 = 0.0
                        local_qt1 = local_qt1 + qf_1
                        local_qt2 = local_qt2 + qf_2
                    else:
                        act = x < 0
                        local_qt1 = local_qt1 + D * (jv_c * Jaref_c) * act
                        local_qt2 = local_qt2 + D * (0.5 * jv_c * jv_c) * act
                    i_c += _K

                # Reduce qt1 and qt2
                sh_val[tid] = local_qt1
                sh_val2[tid] = local_qt2
                qd.simt.block.sync()
                stride = _K // 2
                while stride > 0:
                    if tid < stride:
                        sh_val[tid] += sh_val[tid + stride]
                        sh_val2[tid] += sh_val2[tid + stride]
                    qd.simt.block.sync()
                    stride //= 2

                if tid == 0:
                    qt1_total = constraint_state.quad_gauss[1, i_b] + constraint_state.eq_sum[1, i_b] + sh_val[0]
                    qt2_total = constraint_state.quad_gauss[2, i_b] + constraint_state.eq_sum[2, i_b] + sh_val2[0]
                    g_best = 2.0 * alpha_eval * qt2_total + qt1_total

                    if qd.abs(g_best) > gtol:
                        hess_best = 2.0 * qt2_total
                        newton_done = False

                        # Try one Newton correction first (O(1) compute + 1 cost eval)
                        if hess_best > rigid_global_info.EPS[None]:
                            alpha_nc = alpha_eval - g_best / hess_best
                            if alpha_nc > 0.0:
                                c_nc, g_nc = _ls_eval_cost_grad(alpha_nc, i_b, constraint_state)
                                if c_nc < p0_cost and c_nc < constraint_state.candidates[4, i_b]:
                                    constraint_state.candidates[0, i_b] = alpha_nc
                                    constraint_state.candidates[4, i_b] = c_nc
                                    newton_done = True
                        # Fall back to bisection if Newton didn't converge
                        if not newton_done:
                            bis_a = alpha_eval * 0.5
                            bis_b = alpha_eval
                            if g_best < 0.0:
                                bis_a = alpha_eval
                                bis_b = alpha_eval * 2.0

                            _, g_a = _ls_eval_cost_grad(bis_a, i_b, constraint_state)
                            _, g_b = _ls_eval_cost_grad(bis_b, i_b, constraint_state)

                            if g_a < 0.0 and g_b > 0.0:
                                _N_BISECT = qd.static(LS_BISECT_STEPS)
                                for _bis_it in range(_N_BISECT):
                                    mid_b = (bis_a + bis_b) * 0.5
                                    c_mid_b, g_mid_b = _ls_eval_cost_grad(mid_b, i_b, constraint_state)
                                    if qd.abs(g_mid_b) < gtol or qd.abs(bis_b - bis_a) < rigid_global_info.EPS[None]:
                                        break
                                    if g_mid_b < 0.0:
                                        bis_a = mid_b
                                    else:
                                        bis_b = mid_b
                                mid_b = (bis_a + bis_b) * 0.5
                                c_mid_b, _ = _ls_eval_cost_grad(mid_b, i_b, constraint_state)
                                if c_mid_b < p0_cost and c_mid_b < constraint_state.candidates[4, i_b]:
                                    constraint_state.candidates[0, i_b] = mid_b
                                    constraint_state.candidates[4, i_b] = c_mid_b
        else:
            if tid == 0:
                constraint_state.candidates[0, i_b] = 0.0
            qd.simt.block.sync()

        # === Phase 4: Cooperative apply alpha (fused, saves 1 kernel launch) ===
        qd.simt.block.sync()
        if active:
            n_dofs_apply = constraint_state.qacc.shape[0]
            n_con_apply = constraint_state.n_constraints[i_b]
            alpha_apply = constraint_state.candidates[0, i_b]
            if qd.abs(alpha_apply) < rigid_global_info.EPS[None]:
                if tid == 0:
                    constraint_state.improved[i_b] = False
            else:
                # Apply to dofs (strided over threads)
                i_d = tid
                while i_d < n_dofs_apply:
                    constraint_state.qacc[i_d, i_b] += constraint_state.search[i_d, i_b] * alpha_apply
                    constraint_state.Ma[i_d, i_b] += constraint_state.mv[i_d, i_b] * alpha_apply
                    i_d += _K
                # Apply to constraints (strided over threads)
                i_c = tid
                while i_c < n_con_apply:
                    constraint_state.Jaref[i_c, i_b] += constraint_state.jv[i_c, i_b] * alpha_apply
                    i_c += _K


# ============================================== Shared iteration funcs ================================================


@qd.func
def _func_cg_only_save_prev_grad(
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    """Save prev_grad and prev_Mgrad (CG only)"""
    _B = constraint_state.grad.shape[1]
    qd.loop_config(
        name="cg_only_save_prev_grag", serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32
    )
    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            solver.func_save_prev_grad(i_b, constraint_state=constraint_state)


@qd.func
def _func_update_constraint_forces(
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    """Compute active flags and efc_force, parallelized over (constraint, env)."""
    len_constraints = constraint_state.active.shape[0]
    _B = constraint_state.grad.shape[1]

    qd.loop_config(name="update_constraint_forces")
    for i_c, i_b in qd.ndrange(len_constraints, _B):
        if i_c < constraint_state.n_constraints[i_b] and constraint_state.improved[i_b]:
            ne = constraint_state.n_constraints_equality[i_b]
            nef = ne + constraint_state.n_constraints_frictionloss[i_b]

            if qd.static(static_rigid_sim_config.solver_type == gs.constraint_solver.Newton):
                constraint_state.prev_active[i_c, i_b] = constraint_state.active[i_c, i_b]

            constraint_state.active[i_c, i_b] = True
            floss_force = gs.qd_float(0.0)

            if ne <= i_c and i_c < nef:
                f = constraint_state.efc_frictionloss[i_c, i_b]
                r = constraint_state.diag[i_c, i_b]
                rf = r * f
                linear_neg = constraint_state.Jaref[i_c, i_b] <= -rf
                linear_pos = constraint_state.Jaref[i_c, i_b] >= rf
                constraint_state.active[i_c, i_b] = not (linear_neg or linear_pos)
                floss_force = linear_neg * f + linear_pos * -f
            elif nef <= i_c:
                constraint_state.active[i_c, i_b] = constraint_state.Jaref[i_c, i_b] < 0

            constraint_state.efc_force[i_c, i_b] = floss_force + (
                -constraint_state.Jaref[i_c, i_b] * constraint_state.efc_D[i_c, i_b] * constraint_state.active[i_c, i_b]
            )


@qd.func
def _func_update_constraint_qfrc(
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    """Compute qfrc_constraint = J^T @ efc_force, parallelized over (dof, env)."""
    n_dofs = constraint_state.qfrc_constraint.shape[0]
    _B = constraint_state.grad.shape[1]

    qd.loop_config(name="update_constraint_qfrc")
    for i_d, i_b in qd.ndrange(n_dofs, _B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            n_con = constraint_state.n_constraints[i_b]
            qfrc = gs.qd_float(0.0)
            for i_c in range(n_con):
                qfrc += constraint_state.jac[i_c, i_d, i_b] * constraint_state.efc_force[i_c, i_b]
            constraint_state.qfrc_constraint[i_d, i_b] = qfrc


@qd.func
def _func_update_constraint_cost(
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
):
    """Compute gauss and cost (reductions over dofs and constraints). One thread per env."""
    _B = constraint_state.grad.shape[1]

    qd.loop_config(name="update_constraint_cost", block_dim=32)
    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            n_dofs = constraint_state.qfrc_constraint.shape[0]
            ne = constraint_state.n_constraints_equality[i_b]
            nef = ne + constraint_state.n_constraints_frictionloss[i_b]
            n_con = constraint_state.n_constraints[i_b]

            constraint_state.prev_cost[i_b] = constraint_state.cost[i_b]

            cost_i = gs.qd_float(0.0)
            gauss_i = gs.qd_float(0.0)

            # Gauss cost from dofs
            for i_d in range(n_dofs):
                v = (
                    0.5
                    * (constraint_state.Ma[i_d, i_b] - dofs_state.force[i_d, i_b])
                    * (constraint_state.qacc[i_d, i_b] - dofs_state.acc_smooth[i_d, i_b])
                )
                gauss_i += v
                cost_i += v

            # Constraint cost: quadratic + friction linear
            for i_c in range(n_con):
                cost_i += 0.5 * (
                    constraint_state.Jaref[i_c, i_b] ** 2
                    * constraint_state.efc_D[i_c, i_b]
                    * constraint_state.active[i_c, i_b]
                )
                if ne <= i_c and i_c < nef:
                    f = constraint_state.efc_frictionloss[i_c, i_b]
                    r = constraint_state.diag[i_c, i_b]
                    rf = r * f
                    linear_neg = constraint_state.Jaref[i_c, i_b] <= -rf
                    linear_pos = constraint_state.Jaref[i_c, i_b] >= rf
                    cost_i += linear_neg * f * (-0.5 * rf - constraint_state.Jaref[i_c, i_b]) + linear_pos * f * (
                        -0.5 * rf + constraint_state.Jaref[i_c, i_b]
                    )

            constraint_state.gauss[i_b] = gauss_i
            constraint_state.cost[i_b] = cost_i


@qd.func
def _func_newton_only_nt_hessian(
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    """Step 4: Newton Hessian update (Newton only)"""
    solver.func_hessian_direct_tiled(constraint_state=constraint_state, rigid_global_info=rigid_global_info)
    if qd.static(static_rigid_sim_config.enable_tiled_cholesky_hessian):
        solver.func_cholesky_factor_direct_tiled(
            constraint_state=constraint_state,
            rigid_global_info=rigid_global_info,
            static_rigid_sim_config=static_rigid_sim_config,
        )
    else:
        _B = constraint_state.jac.shape[2]
        qd.loop_config(
            name="cholesky_factor_direct_batch",
            serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL,
            block_dim=32,
        )
        for i_b in range(_B):
            if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
                solver.func_cholesky_factor_direct_batch(
                    i_b=i_b, constraint_state=constraint_state, rigid_global_info=rigid_global_info
                )


@qd.func
def _func_update_gradient(
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    """Step 5: Update gradient"""
    _B = constraint_state.grad.shape[1]
    qd.loop_config(
        name="update_gradient", serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32
    )
    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            solver.func_update_gradient_batch(
                i_b,
                dofs_state=dofs_state,
                entities_info=entities_info,
                rigid_global_info=rigid_global_info,
                constraint_state=constraint_state,
                static_rigid_sim_config=static_rigid_sim_config,
            )


@qd.func
def _func_update_search_direction(
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
):
    """Step 6: Check convergence and update search direction"""
    _B = constraint_state.grad.shape[1]
    qd.loop_config(
        name="update_search_direction", serialize=static_rigid_sim_config.para_level < gs.PARA_LEVEL.ALL, block_dim=32
    )
    for i_b in range(_B):
        if constraint_state.n_constraints[i_b] > 0 and constraint_state.improved[i_b]:
            solver.func_terminate_or_update_descent_batch(
                i_b,
                rigid_global_info=rigid_global_info,
                constraint_state=constraint_state,
                static_rigid_sim_config=static_rigid_sim_config,
            )


@qd.func
def _func_check_early_exit(
    constraint_state: array_class.ConstraintState,
    graph_counter: qd.types.ndarray(qd.i32, ndim=0),
):
    """Decrement iteration counter and exit early if no batch element improved."""
    qd.loop_config(name="check_early_exit_reset_flag")
    for _ in range(1):
        graph_counter[()] = graph_counter[()] - 1
        constraint_state.early_exit_flag[()] = 0

    _B = constraint_state.grad.shape[1]
    qd.loop_config(name="check_early_exit_scan_values")
    for i_b in range(_B):
        if constraint_state.improved[i_b]:
            qd.atomic_max(constraint_state.early_exit_flag[()], 1)

    qd.loop_config(name="check_early_exit_set_counter")
    for _ in range(1):
        if constraint_state.early_exit_flag[()] == 0:
            graph_counter[()] = 0


# ============================================== Solve body dispatch ================================================


@qd.kernel(gpu_graph=True, fastcache=gs.use_fastcache)
def _kernel_solve_gpu_graph(
    dofs_info: array_class.DofsInfo,
    entities_info: array_class.EntitiesInfo,
    dofs_state: array_class.DofsState,
    constraint_state: array_class.ConstraintState,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: qd.template(),
    graph_counter: qd.types.ndarray(qd.i32, ndim=0),
):
    while qd.graph_do_while(graph_counter):
        _func_iterative_linesearch(
            dofs_info, entities_info, dofs_state, constraint_state, rigid_global_info, static_rigid_sim_config
        )
        if qd.static(static_rigid_sim_config.solver_type == gs.constraint_solver.CG):
            _func_cg_only_save_prev_grad(constraint_state, static_rigid_sim_config)
        _func_update_constraint_forces(constraint_state, static_rigid_sim_config)
        _func_update_constraint_qfrc(constraint_state, static_rigid_sim_config)
        _func_update_constraint_cost(dofs_state, constraint_state, static_rigid_sim_config)
        if qd.static(static_rigid_sim_config.solver_type == gs.constraint_solver.Newton):
            _func_newton_only_nt_hessian(constraint_state, rigid_global_info, static_rigid_sim_config)
        _func_update_gradient(entities_info, dofs_state, constraint_state, rigid_global_info, static_rigid_sim_config)
        _func_update_search_direction(constraint_state, rigid_global_info, static_rigid_sim_config)
        _func_check_early_exit(constraint_state, graph_counter)


@solver.func_solve_body.register(
    is_compatible=lambda *args, **kwargs: (
        not (static_rigid_sim_config := solver._get_static_config(*args, **kwargs)).requires_grad
        and static_rigid_sim_config.prefer_parallel_linesearch != 0
    )
)
def func_solve_decomposed(
    entities_info,
    dofs_info,
    dofs_state,
    constraint_state,
    rigid_global_info,
    static_rigid_sim_config,
    _n_iterations,
):
    """
    GPU graph accelerated solver loop with iterative bracket-search linesearch and GPU-side iteration via
    graph_do_while.

    On CUDA SM 9.0+ (Hopper), the entire iteration loop runs on the GPU with no host involvement. On older CUDA GPUs,
    falls back to a host-side do-while loop that still benefits from CUDA graph kernel launch batching. On other GPUs,
    falls back to a host-side C++-side loop, that still reduces python launch overhead.

    Early exits when all batch elements have converged (no improved[i_b] is True).
    """
    if _n_iterations <= 0:
        return
    constraint_state.graph_counter.from_numpy(np.array(_n_iterations, dtype=np.int32))
    _kernel_solve_gpu_graph(
        dofs_info,
        entities_info,
        dofs_state,
        constraint_state,
        rigid_global_info,
        static_rigid_sim_config,
        constraint_state.graph_counter,
    )
