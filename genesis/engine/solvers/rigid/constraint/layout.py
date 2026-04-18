"""Constraint-state layout-switchable accessors.

The fields ``Jaref``, ``jv``, ``efc_D``, ``efc_frictionloss``, ``diag`` and ``active`` are
allocated either as ``(len_constraints_, _B)`` (default) or ``(_B, len_constraints_)``
(when ``static_rigid_sim_config.constraint_layout_transposed`` is ``True``). The accessors
below pick the index order at trace time via ``qd.static(...)`` and are inlined into the
calling kernel — zero runtime overhead either way.

This pattern is fastcache-safe because the flag lives on a kernel argument
(``static_rigid_sim_config``), not a Python global. See ``perso_hugh/doc/linesearch_shuffle.md``.
"""

import quadrants as qd

import genesis.utils.array_class as array_class


# -------- Jaref --------


@qd.func
def get_Jaref(
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
    i_c: qd.int32,
    i_b: qd.int32,
):
    if qd.static(static_rigid_sim_config.constraint_layout_transposed):
        return constraint_state.Jaref[i_b, i_c]
    else:
        return constraint_state.Jaref[i_c, i_b]


@qd.func
def set_Jaref(
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
    i_c: qd.int32,
    i_b: qd.int32,
    value,
):
    if qd.static(static_rigid_sim_config.constraint_layout_transposed):
        constraint_state.Jaref[i_b, i_c] = value
    else:
        constraint_state.Jaref[i_c, i_b] = value


@qd.func
def add_Jaref(
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
    i_c: qd.int32,
    i_b: qd.int32,
    delta,
):
    if qd.static(static_rigid_sim_config.constraint_layout_transposed):
        constraint_state.Jaref[i_b, i_c] += delta
    else:
        constraint_state.Jaref[i_c, i_b] += delta


# -------- jv --------


@qd.func
def get_jv(
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
    i_c: qd.int32,
    i_b: qd.int32,
):
    if qd.static(static_rigid_sim_config.constraint_layout_transposed):
        return constraint_state.jv[i_b, i_c]
    else:
        return constraint_state.jv[i_c, i_b]


@qd.func
def set_jv(
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
    i_c: qd.int32,
    i_b: qd.int32,
    value,
):
    if qd.static(static_rigid_sim_config.constraint_layout_transposed):
        constraint_state.jv[i_b, i_c] = value
    else:
        constraint_state.jv[i_c, i_b] = value


# -------- efc_D --------


@qd.func
def get_efc_D(
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
    i_c: qd.int32,
    i_b: qd.int32,
):
    if qd.static(static_rigid_sim_config.constraint_layout_transposed):
        return constraint_state.efc_D[i_b, i_c]
    else:
        return constraint_state.efc_D[i_c, i_b]


@qd.func
def set_efc_D(
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
    i_c: qd.int32,
    i_b: qd.int32,
    value,
):
    if qd.static(static_rigid_sim_config.constraint_layout_transposed):
        constraint_state.efc_D[i_b, i_c] = value
    else:
        constraint_state.efc_D[i_c, i_b] = value


# -------- efc_frictionloss --------


@qd.func
def get_efc_frictionloss(
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
    i_c: qd.int32,
    i_b: qd.int32,
):
    if qd.static(static_rigid_sim_config.constraint_layout_transposed):
        return constraint_state.efc_frictionloss[i_b, i_c]
    else:
        return constraint_state.efc_frictionloss[i_c, i_b]


@qd.func
def set_efc_frictionloss(
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
    i_c: qd.int32,
    i_b: qd.int32,
    value,
):
    if qd.static(static_rigid_sim_config.constraint_layout_transposed):
        constraint_state.efc_frictionloss[i_b, i_c] = value
    else:
        constraint_state.efc_frictionloss[i_c, i_b] = value


# -------- diag --------


@qd.func
def get_diag(
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
    i_c: qd.int32,
    i_b: qd.int32,
):
    if qd.static(static_rigid_sim_config.constraint_layout_transposed):
        return constraint_state.diag[i_b, i_c]
    else:
        return constraint_state.diag[i_c, i_b]


@qd.func
def set_diag(
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
    i_c: qd.int32,
    i_b: qd.int32,
    value,
):
    if qd.static(static_rigid_sim_config.constraint_layout_transposed):
        constraint_state.diag[i_b, i_c] = value
    else:
        constraint_state.diag[i_c, i_b] = value


# -------- active --------


@qd.func
def get_active(
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
    i_c: qd.int32,
    i_b: qd.int32,
):
    if qd.static(static_rigid_sim_config.constraint_layout_transposed):
        return constraint_state.active[i_b, i_c]
    else:
        return constraint_state.active[i_c, i_b]


@qd.func
def set_active(
    constraint_state: array_class.ConstraintState,
    static_rigid_sim_config: qd.template(),
    i_c: qd.int32,
    i_b: qd.int32,
    value,
):
    if qd.static(static_rigid_sim_config.constraint_layout_transposed):
        constraint_state.active[i_b, i_c] = value
    else:
        constraint_state.active[i_c, i_b] = value
