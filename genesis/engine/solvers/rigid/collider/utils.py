import quadrants as qd

import genesis as gs
import genesis.utils.array_class as array_class
from genesis.constants import GEOM_TYPE


@qd.func
def func_closest_points_on_segments(
    seg_a_p1,
    seg_a_p2,
    seg_b_p1,
    seg_b_p2,
    EPS,
):
    """
    Compute closest points on two line segments using analytical solution.

    References
    ----------
    Real-Time Collision Detection by Christer Ericson, Chapter 5.1.9
    """
    segment_a_dir = seg_a_p2 - seg_a_p1
    segment_b_dir = seg_b_p2 - seg_b_p1
    vec_between_segment_origins = seg_a_p1 - seg_b_p1

    a_squared_len = segment_a_dir.dot(segment_a_dir)
    dot_product_dir = segment_a_dir.dot(segment_b_dir)
    b_squared_len = segment_b_dir.dot(segment_b_dir)
    d = segment_a_dir.dot(vec_between_segment_origins)
    e = segment_b_dir.dot(vec_between_segment_origins)

    denom = a_squared_len * b_squared_len - dot_product_dir * dot_product_dir

    s = gs.qd_float(0.0)
    t = gs.qd_float(0.0)

    if denom < EPS:
        # Segments are parallel or one/both are degenerate
        s = 0.0
        if b_squared_len > EPS:
            t = qd.math.clamp(e / b_squared_len, 0.0, 1.0)
        else:
            t = 0.0
    else:
        # General case: solve for optimal parameters
        s = (dot_product_dir * e - b_squared_len * d) / denom
        t = (a_squared_len * e - dot_product_dir * d) / denom

        s = qd.math.clamp(s, 0.0, 1.0)

        # Recompute t for clamped s
        t = qd.math.clamp((dot_product_dir * s + e) / b_squared_len if b_squared_len > EPS else 0.0, 0.0, 1.0)

        # Recompute s for clamped t (ensures we're on segment boundaries)
        s_new = qd.math.clamp((dot_product_dir * t - d) / a_squared_len if a_squared_len > EPS else 0.0, 0.0, 1.0)

        # Use refined s if it improves the solution
        if a_squared_len > EPS:
            s = s_new

    seg_a_closest = seg_a_p1 + s * segment_a_dir
    seg_b_closest = seg_b_p1 + t * segment_b_dir

    return seg_a_closest, seg_b_closest


@qd.func
def func_det3(
    v1,
    v2,
    v3,
):
    """
    Compute the determinant of a 3x3 matrix M = [v1 | v2 | v3].
    """
    return (
        v1[0] * (v2[1] * v3[2] - v2[2] * v3[1])
        - v1[1] * (v2[0] * v3[2] - v2[2] * v3[0])
        + v1[2] * (v2[0] * v3[1] - v2[1] * v3[0])
    )


@qd.func
def func_point_in_geom_aabb(
    geoms_state: array_class.GeomsState,
    i_g: qd.i32,
    i_b: qd.i32,
    point: qd.types.vector(3, qd.f32),
    expansion: qd.f32 = 0.0,
):
    aabb_min = geoms_state.aabb_min[i_g, i_b] - expansion
    aabb_max = geoms_state.aabb_max[i_g, i_b] + expansion
    return (point > aabb_min).all() and (point < aabb_max).all()


@qd.func
def func_is_geom_aabbs_overlap(geoms_state: array_class.GeomsState, i_ga, i_gb, i_b):
    return not (
        (geoms_state.aabb_max[i_ga, i_b] <= geoms_state.aabb_min[i_gb, i_b]).any()
        or (geoms_state.aabb_min[i_ga, i_b] >= geoms_state.aabb_max[i_gb, i_b]).any()
    )


@qd.func
def func_is_obbs_overlap(
    geoms_state: array_class.GeomsState,
    collider_info: array_class.ColliderInfo,
    i_ga,
    i_gb,
    i_b,
):
    """6-axis SAT oriented bounding box overlap test (Gottschalk et al.).

    Tests the 3 face normals of each OBB as separating axes.  Uses a flag
    (no early return) to comply with Quadrants JIT constraints.
    """
    size_a = collider_info.geom_obb_halfsize[i_ga]
    size_b = collider_info.geom_obb_halfsize[i_gb]
    center_a = collider_info.geom_obb_center[i_ga]
    center_b = collider_info.geom_obb_center[i_gb]

    pos_a = geoms_state.pos[i_ga, i_b]
    pos_b = geoms_state.pos[i_gb, i_b]
    qa = geoms_state.quat[i_ga, i_b]
    qb = geoms_state.quat[i_gb, i_b]

    # Rotation matrix columns from quaternion (w, x, y, z)
    wa = qa[0]; xa = qa[1]; ya = qa[2]; za = qa[3]
    a0 = qd.Vector([1.0 - 2.0 * (ya * ya + za * za), 2.0 * (xa * ya + wa * za), 2.0 * (xa * za - wa * ya)], dt=gs.qd_float)
    a1 = qd.Vector([2.0 * (xa * ya - wa * za), 1.0 - 2.0 * (xa * xa + za * za), 2.0 * (ya * za + wa * xa)], dt=gs.qd_float)
    a2 = qd.Vector([2.0 * (xa * za + wa * ya), 2.0 * (ya * za - wa * xa), 1.0 - 2.0 * (xa * xa + ya * ya)], dt=gs.qd_float)

    wb = qb[0]; xb = qb[1]; yb = qb[2]; zb = qb[3]
    b0 = qd.Vector([1.0 - 2.0 * (yb * yb + zb * zb), 2.0 * (xb * yb + wb * zb), 2.0 * (xb * zb - wb * yb)], dt=gs.qd_float)
    b1 = qd.Vector([2.0 * (xb * yb - wb * zb), 1.0 - 2.0 * (xb * xb + zb * zb), 2.0 * (yb * zb + wb * xb)], dt=gs.qd_float)
    b2 = qd.Vector([2.0 * (xb * zb + wb * yb), 2.0 * (yb * zb - wb * xb), 1.0 - 2.0 * (xb * xb + yb * yb)], dt=gs.qd_float)

    # World centers: R @ local_center + world_pos
    wc_a = a0 * center_a[0] + a1 * center_a[1] + a2 * center_a[2] + pos_a
    wc_b = b0 * center_b[0] + b1 * center_b[1] + b2 * center_b[2] + pos_b
    d = wc_b - wc_a

    # Compute max separation across 6 axes; positive means separated.
    sep = gs.qd_float(-1e30)

    # A's 3 face normals
    gap = qd.abs(d.dot(a0)) - size_a[0] - (qd.abs(size_b[0] * b0.dot(a0)) + qd.abs(size_b[1] * b1.dot(a0)) + qd.abs(size_b[2] * b2.dot(a0)))
    sep = qd.max(sep, gap)

    gap = qd.abs(d.dot(a1)) - size_a[1] - (qd.abs(size_b[0] * b0.dot(a1)) + qd.abs(size_b[1] * b1.dot(a1)) + qd.abs(size_b[2] * b2.dot(a1)))
    sep = qd.max(sep, gap)

    gap = qd.abs(d.dot(a2)) - size_a[2] - (qd.abs(size_b[0] * b0.dot(a2)) + qd.abs(size_b[1] * b1.dot(a2)) + qd.abs(size_b[2] * b2.dot(a2)))
    sep = qd.max(sep, gap)

    # B's 3 face normals
    gap = qd.abs(d.dot(b0)) - size_b[0] - (qd.abs(size_a[0] * a0.dot(b0)) + qd.abs(size_a[1] * a1.dot(b0)) + qd.abs(size_a[2] * a2.dot(b0)))
    sep = qd.max(sep, gap)

    gap = qd.abs(d.dot(b1)) - size_b[1] - (qd.abs(size_a[0] * a0.dot(b1)) + qd.abs(size_a[1] * a1.dot(b1)) + qd.abs(size_a[2] * a2.dot(b1)))
    sep = qd.max(sep, gap)

    gap = qd.abs(d.dot(b2)) - size_b[2] - (qd.abs(size_a[0] * a0.dot(b2)) + qd.abs(size_a[1] * a1.dot(b2)) + qd.abs(size_a[2] * a2.dot(b2)))
    sep = qd.max(sep, gap)

    return sep <= 0.0


@qd.func
def func_is_discrete_geom(
    geoms_info: array_class.GeomsInfo,
    i_g,
):
    """
    Check if the given geom is a discrete geometry.
    """
    geom_type = geoms_info.type[i_g]
    return geom_type == GEOM_TYPE.MESH or geom_type == GEOM_TYPE.BOX


@qd.func
def func_is_discrete_geoms(
    geoms_info: array_class.GeomsInfo,
    i_ga,
    i_gb,
):
    """
    Check if the given geoms are discrete geometries.
    """
    return func_is_discrete_geom(geoms_info, i_ga) and func_is_discrete_geom(geoms_info, i_gb)


@qd.func
def func_is_equal_vec(a, b, eps):
    """
    Check if two vectors are equal within a small tolerance.
    """
    return (qd.abs(a - b) < eps).all()
