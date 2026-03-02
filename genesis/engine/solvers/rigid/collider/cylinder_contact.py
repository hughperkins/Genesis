import quadrants as qd

import genesis as gs
import genesis.utils.geom as gu
from . import utils


@qd.func
def func_sphere_sphere_contact(
    ga_pos,
    gb_pos,
    radius_a: gs.qd_float,
    radius_b: gs.qd_float,
    EPS,
):
    """Analytical sphere-sphere collision detection.

    Returns (is_col, normal, contact_pos, penetration).
    Contact normal points from B to A.
    """
    diff = ga_pos - gb_pos
    dist_sq = diff.dot(diff)
    combined_radius = radius_a + radius_b

    is_col = False
    normal = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)
    contact_pos = qd.Vector.zero(gs.qd_float, 3)
    penetration = gs.qd_float(0.0)

    if dist_sq < combined_radius * combined_radius:
        dist = qd.sqrt(dist_sq)
        is_col = True
        if dist > EPS:
            normal = diff / dist
        penetration = combined_radius - dist
        contact_pos = ga_pos - (radius_a - 0.5 * penetration) * normal

    return is_col, normal, contact_pos, penetration


@qd.func
def _closest_point_on_cylinder(
    cyl_pos,
    cyl_quat,
    cyl_radius: gs.qd_float,
    cyl_halflength: gs.qd_float,
    query_point,
    EPS,
):
    """Find the closest point on a cylinder surface (barrel + flat caps) to a query point.

    Returns (closest_point, is_on_cap, is_inside).  The cylinder is oriented along local Z.
    is_inside is True when the query point lies within the cylinder volume.
    """
    local_z = qd.Vector([0.0, 0.0, 1.0], dt=gs.qd_float)
    axis = gu.qd_transform_by_quat_fast(local_z, cyl_quat)

    delta = query_point - cyl_pos
    axial_proj = delta.dot(axis)
    radial_vec = delta - axial_proj * axis
    radial_dist = qd.sqrt(radial_vec.dot(radial_vec))

    is_on_cap = False
    is_inside = False
    closest = qd.Vector.zero(gs.qd_float, 3)
    perp = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)

    radial_dir = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)
    has_radial = radial_dist > EPS

    if has_radial:
        radial_dir = radial_vec / radial_dist
    else:
        if qd.abs(axis[0]) < 0.9:
            perp = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)
        else:
            perp = qd.Vector([0.0, 1.0, 0.0], dt=gs.qd_float)
        radial_dir = perp - perp.dot(axis) * axis
        radial_dir = radial_dir / qd.sqrt(radial_dir.dot(radial_dir) + 1e-30)

    beyond_cap = qd.abs(axial_proj) > cyl_halflength

    if beyond_cap:
        cap_sign = gs.qd_float(1.0)
        if axial_proj < 0.0:
            cap_sign = gs.qd_float(-1.0)
        cap_center = cyl_pos + cap_sign * cyl_halflength * axis
        is_on_cap = True
        if radial_dist <= cyl_radius:
            closest = cap_center + radial_dist * radial_dir
        else:
            closest = cap_center + cyl_radius * radial_dir
    else:
        axis_point = cyl_pos + axial_proj * axis
        if radial_dist <= cyl_radius:
            is_inside = True
            barrel_dist = cyl_radius - radial_dist
            cap_dist = cyl_halflength - qd.abs(axial_proj)
            if barrel_dist < cap_dist:
                closest = axis_point + cyl_radius * radial_dir
            else:
                cap_sign = gs.qd_float(1.0)
                if axial_proj < 0.0:
                    cap_sign = gs.qd_float(-1.0)
                closest = cyl_pos + cap_sign * cyl_halflength * axis + radial_dist * radial_dir
                is_on_cap = True
        else:
            closest = axis_point + cyl_radius * radial_dir

    return closest, is_on_cap, is_inside


@qd.func
def func_cylinder_sphere_contact(
    cyl_pos,
    cyl_quat,
    cyl_radius: gs.qd_float,
    cyl_halflength: gs.qd_float,
    sph_pos,
    sph_radius: gs.qd_float,
    normal_sign: gs.qd_int,
    EPS,
):
    """Analytical cylinder-sphere collision detection.

    Returns (is_col, normal, contact_pos, penetration).
    normal_sign should be +1 if i_ga is the cylinder, -1 if i_gb is.
    """
    closest, _is_on_cap, is_inside = _closest_point_on_cylinder(
        cyl_pos, cyl_quat, cyl_radius, cyl_halflength, sph_pos, EPS
    )

    diff = sph_pos - closest
    dist_sq = diff.dot(diff)

    is_col = False
    normal = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)
    contact_pos = qd.Vector.zero(gs.qd_float, 3)
    penetration = gs.qd_float(0.0)

    if dist_sq < sph_radius * sph_radius:
        dist = qd.sqrt(dist_sq)
        is_col = True
        if dist > EPS:
            normal = diff / dist
        penetration = sph_radius - dist
        # contact_pos formula works correctly with the raw normal for both cases
        contact_pos = sph_pos - (sph_radius - 0.5 * penetration) * normal
        # When the sphere center is inside the cylinder volume, diff points
        # inward (from surface toward interior). Flip to get outward normal.
        if is_inside:
            normal = -normal
        normal = normal * gs.qd_float(normal_sign)

    return is_col, normal, contact_pos, penetration


@qd.func
def _cap_vs_barrel(
    cap_center,
    cap_normal,
    cap_radius: gs.qd_float,
    barrel_pos,
    barrel_axis,
    barrel_radius: gs.qd_float,
    barrel_halflength: gs.qd_float,
    EPS,
):
    """Contact between a flat cap disc and a cylinder barrel.

    The cap is a disc at *cap_center* with outward normal *cap_normal*
    and radius *cap_radius*.  The barrel is the cylindrical surface at
    distance *barrel_radius* from the line through *barrel_pos* along
    *barrel_axis*, extending ±*barrel_halflength*.

    Returns (is_col, normal, contact_pos, penetration) where the
    returned normal equals the cap outward normal.
    """
    is_col = False
    out_normal = cap_normal
    contact_pos = qd.Vector.zero(gs.qd_float, 3)
    pen = gs.qd_float(0.0)

    a = barrel_axis.dot(cap_normal)
    n_perp = cap_normal - a * barrel_axis
    n_perp_len_sq = n_perp.dot(n_perp)

    if n_perp_len_sq > EPS * EPS:
        n_perp_len = qd.sqrt(n_perp_len_sq)
        d_opt = -n_perp / n_perp_len

        t_opt = gs.qd_float(0.0)
        if qd.abs(a) > EPS:
            t_opt = -barrel_halflength
            if a < gs.qd_float(0.0):
                t_opt = barrel_halflength
        else:
            t_opt = (cap_center - barrel_pos).dot(barrel_axis)
            t_opt = qd.max(-barrel_halflength, qd.min(barrel_halflength, t_opt))

        barrel_pt = barrel_pos + t_opt * barrel_axis + barrel_radius * d_opt
        s = (barrel_pt - cap_center).dot(cap_normal)

        # Use a small margin so that surfaces exactly at the boundary
        # (s == 0) are still handled by cap logic instead of falling
        # through to the barrel-barrel fallback.
        margin = barrel_radius * gs.qd_float(0.02)
        if s < margin:
            proj = barrel_pt - s * cap_normal
            offset = proj - cap_center
            if offset.dot(offset) < cap_radius * cap_radius:
                is_col = True
                pen = qd.max(-s, gs.qd_float(0.0))
                out_normal = cap_normal
                contact_pos = barrel_pt + gs.qd_float(0.5) * pen * cap_normal

    return is_col, out_normal, contact_pos, pen


@qd.func
def func_cylinder_cylinder_contact(
    ga_pos,
    ga_quat,
    radius_a: gs.qd_float,
    halflength_a: gs.qd_float,
    gb_pos,
    gb_quat,
    radius_b: gs.qd_float,
    halflength_b: gs.qd_float,
    EPS,
):
    """Analytical cylinder-cylinder collision detection (single contact).

    Returns (is_col, normal, contact_pos, penetration).
    Normal points from B to A.
    """
    local_z = qd.Vector([0.0, 0.0, 1.0], dt=gs.qd_float)
    axis_a = gu.qd_transform_by_quat_fast(local_z, ga_quat)
    axis_b = gu.qd_transform_by_quat_fast(local_z, gb_quat)

    A1 = ga_pos - halflength_a * axis_a
    A2 = ga_pos + halflength_a * axis_a
    B1 = gb_pos - halflength_b * axis_b
    B2 = gb_pos + halflength_b * axis_b

    combined_radius = radius_a + radius_b
    center_diff = ga_pos - gb_pos

    is_col = False
    normal = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)
    contact_pos = qd.Vector.zero(gs.qd_float, 3)
    penetration = gs.qd_float(0.0)

    arb = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)

    axis_cross = axis_a.cross(axis_b)
    axis_cross_len_sq = axis_cross.dot(axis_cross)

    if axis_cross_len_sq < EPS * EPS:
        # Parallel axes
        perp_vec = center_diff - center_diff.dot(axis_a) * axis_a
        perp_dist = qd.sqrt(perp_vec.dot(perp_vec))

        b_on_a = -center_diff.dot(axis_a)
        overlap_start = qd.max(-halflength_a, b_on_a - halflength_b)
        overlap_end = qd.min(halflength_a, b_on_a + halflength_b)

        if overlap_end > overlap_start and perp_dist < combined_radius:
            is_col = True
            penetration = combined_radius - perp_dist
            if perp_dist > EPS:
                normal = perp_vec / perp_dist
            else:
                if qd.abs(axis_a[0]) < 0.9:
                    arb = qd.Vector([1.0, 0.0, 0.0], dt=gs.qd_float)
                else:
                    arb = qd.Vector([0.0, 1.0, 0.0], dt=gs.qd_float)
                normal = arb - arb.dot(axis_a) * axis_a
                normal = normal / qd.sqrt(normal.dot(normal) + 1e-30)

            overlap_center = 0.5 * (overlap_start + overlap_end)
            radial_offset = -(radius_a - 0.5 * penetration) * normal
            contact_pos = ga_pos + overlap_center * axis_a + radial_offset

        elif perp_dist < combined_radius:
            Pa, Pb = utils.func_closest_points_on_segments(A1, A2, B1, B2, EPS)
            radial = Pa - Pb
            radial_dist = qd.sqrt(radial.dot(radial))
            if radial_dist > EPS and radial_dist < combined_radius:
                is_col = True
                normal = radial / radial_dist
                penetration = combined_radius - radial_dist
                contact_pos = Pb + (radius_b - 0.5 * penetration) * normal
    else:
        Pa, Pb = utils.func_closest_points_on_segments(A1, A2, B1, B2, EPS)
        radial = Pa - Pb
        radial_dist = qd.sqrt(radial.dot(radial))

        s_a = (Pa - ga_pos).dot(axis_a)
        s_b = (Pb - gb_pos).dot(axis_b)
        pa_at_end = (halflength_a - qd.abs(s_a)) < EPS
        pb_at_end = (halflength_b - qd.abs(s_b)) < EPS

        if pa_at_end or pb_at_end:
            # At least one closest point is clamped to a segment endpoint,
            # so the contact involves a flat cap rather than barrel-barrel.
            # Only check the cap at the specific clamped endpoint to avoid
            # phantom contacts from the far cap during deep penetrations.
            best_pen = gs.qd_float(0.0)

            if pb_at_end:
                cap_sign_b = gs.qd_float(1.0)
                if s_b < gs.qd_float(0.0):
                    cap_sign_b = gs.qd_float(-1.0)
                cap_center_b = gb_pos + cap_sign_b * halflength_b * axis_b
                cap_normal_b = cap_sign_b * axis_b
                cb, nb, pb2, db = _cap_vs_barrel(
                    cap_center_b, cap_normal_b, radius_b,
                    ga_pos, axis_a, radius_a, halflength_a, EPS)
                if cb:
                    if db > best_pen:
                        best_pen = db
                        is_col = True
                        normal = nb
                        penetration = db
                        contact_pos = pb2

            if pa_at_end:
                cap_sign_a = gs.qd_float(1.0)
                if s_a < gs.qd_float(0.0):
                    cap_sign_a = gs.qd_float(-1.0)
                cap_center_a = ga_pos + cap_sign_a * halflength_a * axis_a
                cap_normal_a = cap_sign_a * axis_a
                ca, na, pa2, da = _cap_vs_barrel(
                    cap_center_a, cap_normal_a, radius_a,
                    gb_pos, axis_b, radius_b, halflength_b, EPS)
                if ca:
                    if da > best_pen:
                        best_pen = da
                        is_col = True
                        normal = -na
                        penetration = da
                        contact_pos = pa2

            if not is_col:
                if radial_dist > EPS:
                    if radial_dist < combined_radius:
                        is_col = True
                        normal = radial / radial_dist
                        penetration = combined_radius - radial_dist
                        contact_pos = Pb + (radius_b - gs.qd_float(0.5) * penetration) * normal
                else:
                    cross_len = qd.sqrt(axis_cross_len_sq)
                    sep_dir = axis_cross / cross_len
                    if sep_dir.dot(center_diff) < 0.0:
                        sep_dir = -sep_dir
                    is_col = True
                    normal = sep_dir
                    penetration = combined_radius
                    contact_pos = gs.qd_float(0.5) * (Pa + Pb)
        else:
            if radial_dist > EPS:
                if radial_dist < combined_radius:
                    is_col = True
                    normal = radial / radial_dist
                    penetration = combined_radius - radial_dist
                    contact_pos = Pb + (radius_b - gs.qd_float(0.5) * penetration) * normal
            else:
                cross_len = qd.sqrt(axis_cross_len_sq)
                sep_dir = axis_cross / cross_len
                if sep_dir.dot(center_diff) < 0.0:
                    sep_dir = -sep_dir
                is_col = True
                normal = sep_dir
                penetration = combined_radius
                contact_pos = gs.qd_float(0.5) * (Pa + Pb)

    return is_col, normal, contact_pos, penetration
