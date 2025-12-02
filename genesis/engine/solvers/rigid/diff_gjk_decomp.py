import gstaichi as ti
import genesis as gs
import genesis.utils.geom as gu
import genesis.utils.array_class as array_class
import genesis.engine.solvers.rigid.gjk_decomp as GJK


@ti.func
def func_gjk_contact(
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    geoms_init_AABB: array_class.GeomsInitAABB,
    verts_info: array_class.VertsInfo,
    faces_info: array_class.FacesInfo,
    rigid_global_info: array_class.RigidGlobalInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_static_config: ti.template(),
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    support_field_info: array_class.SupportFieldInfo,
    i_ga,
    i_gb,
    i_b,
    pos_tol,
    normal_tol,
):
    func_differentiable_contact(
        geoms_state, gjk_state.diff_contact_input, gjk_info, i_ga, i_gb, i_b, 0, -1.0
    )


@ti.func
def func_extended_epa(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_static_config: ti.template(),
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    support_field_info: array_class.SupportFieldInfo,
    i_ga,
    i_gb,
    i_b,
    max_iter,
):
    """
    Extended version of safe EPA algorithm to find multiple possible contact points for differentiable contact detection.

    While the original safe EPA algorithm finds the farthest boundary face on the Minkowski difference, this function
    can be used to find nearly-farthest boundary faces on the Minkowski difference. When the configurations of the
    objects are slightly perturbed, the farthest boundary face could change to one of the nearly-farthest boundary faces.
    Therefore, we use this function to find such nearly-farthest boundary faces for differentiability.
    """
    tolerance = gjk_info.tolerance[None]
    nearest_i_f = gs.ti_int(-1)

    discrete = GJK.func_is_discrete_geoms(geoms_info, i_ga, i_gb, i_b)
    if discrete:
        # If the objects are discrete, we do not use tolerance.
        tolerance = gs.EPS

    k = 0
    converged = False
    while k < max_iter:
        k += 1

        # Find the polytope face with the smallest distance to the origin
        lower2 = gjk_info.FLOAT_MAX_SQ[None]
        nearest_i_f = -1

        for i in range(gjk_state.polytope.nfaces_map[i_b]):
            i_f = gjk_state.polytope_faces_map[i_b, i]
            if gjk_state.polytope_faces.visited[i_b, i_f] == 1:
                continue

            face_dist2 = gjk_state.polytope_faces.dist2[i_b, i_f]
            if face_dist2 < lower2:
                lower2 = face_dist2
                nearest_i_f = i_f

        if nearest_i_f == -1:
            break

        # Find a new support point w from the nearest face's normal
        lower = ti.sqrt(lower2)
        dir = gjk_state.polytope_faces.normal[i_b, nearest_i_f]
        wi = GJK.func_epa_support(
            geoms_state,
            geoms_info,
            verts_info,
            static_rigid_sim_config,
            collider_state,
            collider_static_config,
            gjk_state,
            gjk_info,
            support_field_info,
            i_ga,
            i_gb,
            i_b,
            dir,
            1.0,
        )
        w = gjk_state.polytope_verts.mink[i_b, wi]

        # The upper bound of depth at k-th iteration
        upper = w.dot(dir)

        # If the upper bound and lower bound are close enough, we can stop the algorithm
        if (upper - lower) < tolerance:
            converged = True
            break

        if discrete:
            repeated = False
            for i in range(gjk_state.polytope.nverts[i_b]):
                if i == wi:
                    continue
                elif (
                    gjk_state.polytope_verts.id1[i_b, i] == gjk_state.polytope_verts.id1[i_b, wi]
                    and gjk_state.polytope_verts.id2[i_b, i] == gjk_state.polytope_verts.id2[i_b, wi]
                ):
                    # The vertex w is already in the polytope, so we do not need to add it again.
                    repeated = True
                    break
            if repeated:
                nearest_i_f = -1
                break

        gjk_state.polytope.horizon_w[i_b] = w

        # Compute horizon
        horizon_flag = GJK.func_epa_horizon(gjk_state, gjk_info, i_b, nearest_i_f)

        if horizon_flag:
            # There was an error in the horizon construction, so the horizon edge is not a closed loop.
            nearest_i_f = -1
            break

        if gjk_state.polytope.horizon_nedges[i_b] < 3:
            # Should not happen, because at least three edges should be in the horizon from one deleted face.
            nearest_i_f = -1
            break

        # Check if the memory space is enough for attaching new faces
        nfaces = gjk_state.polytope.nfaces[i_b]
        nedges = gjk_state.polytope.horizon_nedges[i_b]
        if nfaces + nedges >= gjk_info.polytope_max_faces[None]:
            # If the polytope is full, we cannot insert new faces
            nearest_i_f = -1
            break

        # Attach the new faces
        attach_flag = GJK.RETURN_CODE.SUCCESS
        for i in range(nedges):
            # Face id of the current face to attach
            i_f0 = nfaces + i
            # Face id of the next face to attach
            i_f1 = nfaces + (i + 1) % nedges

            horizon_i_f = gjk_state.polytope_horizon_data.face_idx[i_b, i]
            horizon_i_e = gjk_state.polytope_horizon_data.edge_idx[i_b, i]

            horizon_v1 = gjk_state.polytope_faces.verts_idx[i_b, horizon_i_f][horizon_i_e]
            horizon_v2 = gjk_state.polytope_faces.verts_idx[i_b, horizon_i_f][(horizon_i_e + 1) % 3]

            # Change the adjacent face index of the existing face
            gjk_state.polytope_faces.adj_idx[i_b, horizon_i_f][horizon_i_e] = i_f0

            # Attach the new face.
            # If this if the first face, will be adjacent to the face that will be attached last.
            adj_i_f_0 = i_f0 - 1 if (i > 0) else nfaces + nedges - 1
            adj_i_f_1 = horizon_i_f
            adj_i_f_2 = i_f1

            attach_flag = GJK.func_safe_attach_face_to_polytope(
                gjk_state,
                gjk_info,
                i_b,
                wi,
                horizon_v2,
                horizon_v1,
                adj_i_f_2,  # Previous face id
                adj_i_f_1,
                adj_i_f_0,  # Next face id
            )
            if attach_flag != GJK.RETURN_CODE.SUCCESS:
                # Unrecoverable numerical issue
                break

            # Store face in the map
            nfaces_map = gjk_state.polytope.nfaces_map[i_b]
            gjk_state.polytope_faces_map[i_b, nfaces_map] = i_f0
            gjk_state.polytope_faces.map_idx[i_b, i_f0] = nfaces_map
            gjk_state.polytope.nfaces_map[i_b] += 1

        if attach_flag != GJK.RETURN_CODE.SUCCESS:
            nearest_i_f = -1
            break

        # Clear the horizon data for the next iteration
        gjk_state.polytope.horizon_nedges[i_b] = 0

        if (gjk_state.polytope.nfaces_map[i_b] == 0) or (nearest_i_f == -1):
            # No face candidate left
            nearest_i_f = -1
            break

    if converged:
        # Remove the last vertex from the polytope, because it was added because of this boundary face
        gjk_state.polytope.nverts[i_b] -= 1
    else:
        nearest_i_f = -1

    if nearest_i_f != -1:
        # Nearest face found
        dist2 = gjk_state.polytope_faces.dist2[i_b, nearest_i_f]
        flag = GJK.func_safe_epa_witness(gjk_state, gjk_info, i_ga, i_gb, i_b, nearest_i_f)
        if flag == GJK.RETURN_CODE.SUCCESS:
            gjk_state.n_witness[i_b] = 1
            gjk_state.distance[i_b] = -ti.sqrt(dist2)
        else:
            # Failed to compute witness points, so the objects are not colliding
            gjk_state.n_witness[i_b] = 0
            gjk_state.distance[i_b] = 0.0
            nearest_i_f = -1
    else:
        # No face found, so the objects are not colliding
        gjk_state.n_witness[i_b] = 0
        gjk_state.distance[i_b] = 0.0

    return nearest_i_f, k


@ti.func
def func_add_diff_contact_input(
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    verts_info: array_class.VertsInfo,
    static_rigid_sim_config: ti.template(),
    collider_state: array_class.ColliderState,
    collider_static_config: ti.template(),
    gjk_state: array_class.GJKState,
    gjk_info: array_class.GJKInfo,
    support_field_info: array_class.SupportFieldInfo,
    i_ga,
    i_gb,
    i_b,
    i_f,
):
    """
    Prepare the (non-differentiable) contact data that can be used for computing the differentiable contact data later.
    Therefore, this function is not differentiable.
    """
    n = gjk_state.n_diff_contact_input[i_b]

    i_v1 = gjk_state.polytope_faces.verts_idx[i_b, i_f][0]
    i_v2 = gjk_state.polytope_faces.verts_idx[i_b, i_f][1]
    i_v3 = gjk_state.polytope_faces.verts_idx[i_b, i_f][2]

    # Define the face (possibly) on the boundary of the Minkowski difference in the default configuration
    mink1 = gs.ti_vec3(0.0, 0.0, 0.0)
    mink2 = gs.ti_vec3(0.0, 0.0, 0.0)
    mink3 = gs.ti_vec3(0.0, 0.0, 0.0)

    for i in ti.static(range(3)):
        curr_i_v = i_v1
        if i == 1:
            curr_i_v = i_v2
        elif i == 2:
            curr_i_v = i_v3

        mink = func_compute_minkowski_point(
            geoms_state.pos[i_ga, i_b],
            geoms_state.quat[i_ga, i_b],
            geoms_state.pos[i_gb, i_b],
            geoms_state.quat[i_gb, i_b],
            gjk_state.polytope_verts.local_obj1[i_b, curr_i_v],
            gjk_state.polytope_verts.local_obj2[i_b, curr_i_v],
        )
        if i == 0:
            mink1 = mink
        elif i == 1:
            mink2 = mink
        elif i == 2:
            mink3 = mink

    ### Check validity of this contact. The contact is valid if the contact information could be computed in numerically
    ### stable way in both the forward and backward pass.
    # (a) Check if the face is degenerate.
    normal = func_plane_normal(mink1, mink2, mink3)
    normal_norm = normal.norm()
    is_face_degenerate = normal_norm < gjk_info.diff_contact_min_normal_norm[None]

    # (b) Check if the origin is very close to the face (which means very small penetration depth).
    proj_o = func_project_origin_to_plane(mink1, mink2, mink3, normal)
    origin_dist = proj_o.norm()
    is_origin_close_to_face = origin_dist < gjk_info.diff_contact_min_penetration[None]

    ### Orient the face normal, so that it points to the other side of the origin.
    face_center = (mink1 + mink2 + mink3) / 3.0
    if normal_norm > gjk_info.FLOAT_MIN[None]:
        normal = normal.normalized()
    if normal.dot(face_center) < 0.0:
        normal = -normal

    ### Compute the support point along the face normal.
    obj1, obj2, localpos1, localpos2, id1, id2, mink = GJK.func_support(
        geoms_state,
        geoms_info,
        verts_info,
        static_rigid_sim_config,
        collider_state,
        collider_static_config,
        gjk_state,
        gjk_info,
        support_field_info,
        i_ga,
        i_gb,
        i_b,
        normal,
        False,
    )

    gjk_state.diff_contact_input.local_pos1_a[i_b, n] = gjk_state.polytope_verts.local_obj1[i_b, i_v1]
    gjk_state.diff_contact_input.local_pos1_b[i_b, n] = gjk_state.polytope_verts.local_obj1[i_b, i_v2]
    gjk_state.diff_contact_input.local_pos1_c[i_b, n] = gjk_state.polytope_verts.local_obj1[i_b, i_v3]
    gjk_state.diff_contact_input.local_pos2_a[i_b, n] = gjk_state.polytope_verts.local_obj2[i_b, i_v1]
    gjk_state.diff_contact_input.local_pos2_b[i_b, n] = gjk_state.polytope_verts.local_obj2[i_b, i_v2]
    gjk_state.diff_contact_input.local_pos2_c[i_b, n] = gjk_state.polytope_verts.local_obj2[i_b, i_v3]
    gjk_state.diff_contact_input.w_local_pos1[i_b, n] = localpos1
    gjk_state.diff_contact_input.w_local_pos2[i_b, n] = localpos2
    gjk_state.diff_contact_input.valid[i_b, n] = not (is_face_degenerate or is_origin_close_to_face)
    gjk_state.n_diff_contact_input[i_b] += 1


@ti.func
def func_contact_orthogonals(
    i_ga,
    i_gb,
    normal: ti.types.vector(3),
    i_b,
    links_state: array_class.LinksState,
    links_info: array_class.LinksInfo,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
    geoms_init_AABB: array_class.GeomsInitAABB,
    rigid_global_info: array_class.RigidGlobalInfo,
):
    EPS = rigid_global_info.EPS[None]

    axis_0 = ti.Vector.zero(gs.ti_float, 3)
    axis_1 = ti.Vector.zero(gs.ti_float, 3)

    # The reference geometry is the one that will have the largest impact on the position of
    # the contact point. Basically, the smallest one between the two, which can be approximated
    # by the volume of their respective bounding box.
    i_g = i_gb
    if geoms_info.type[i_ga] != gs.GEOM_TYPE.PLANE:
        size_ga = geoms_init_AABB[i_ga, 7]
        volume_ga = size_ga[0] * size_ga[1] * size_ga[2]
        size_gb = geoms_init_AABB[i_gb, 7]
        volume_gb = size_gb[0] * size_gb[1] * size_gb[2]
        i_g = i_ga if volume_ga < volume_gb else i_gb

    # Compute orthogonal basis mixing principal inertia axes of geometry with contact normal
    i_l = geoms_info.link_idx[i_g]
    rot = gu.ti_quat_to_R(links_state.i_quat[i_l, i_b], EPS)
    axis_idx = gs.ti_int(0)
    axis_angle_max = gs.ti_float(0.0)
    for i in ti.static(range(3)):
        axis_angle = ti.abs(rot[:, i].dot(normal))
        if axis_angle > axis_angle_max:
            axis_angle_max = axis_angle
            axis_idx = i
    axis_idx = (axis_idx + 1) % 3
    axis_0 = rot[:, axis_idx]
    axis_0 = (axis_0 - normal.dot(axis_0) * normal).normalized()
    axis_1 = normal.cross(axis_0)

    return axis_0, axis_1


@ti.func
def func_rotate_frame(
    i_g,
    contact_pos: ti.types.vector(3),
    qrot: ti.types.vector(4),
    i_b,
    geoms_state: array_class.GeomsState,
    geoms_info: array_class.GeomsInfo,
):
    geoms_state.quat[i_g, i_b] = gu.ti_transform_quat_by_quat(geoms_state.quat[i_g, i_b], qrot)

    rel = contact_pos - geoms_state.pos[i_g, i_b]
    vec = gu.ti_transform_by_quat(rel, qrot)
    vec = vec - rel
    geoms_state.pos[i_g, i_b] = geoms_state.pos[i_g, i_b] - vec


@ti.func
def func_compute_minkowski_point(
    ga_pos: ti.types.vector(3),
    ga_quat: ti.types.vector(4),
    gb_pos: ti.types.vector(3),
    gb_quat: ti.types.vector(4),
    va: ti.types.vector(3),
    vb: ti.types.vector(3),
):
    # Transform the points to the global frame
    va_ = gu.ti_transform_by_trans_quat(va, ga_pos, ga_quat)
    vb_ = gu.ti_transform_by_trans_quat(vb, gb_pos, gb_quat)
    return va_ - vb_


# ------------------------------ Differentiable functions ------------------------------------
# These functions have minimal number of branches to align backward pass with forward pass.
# --------------------------------------------------------------------------------------------
@ti.func
def func_differentiable_contact(
    geoms_state: array_class.GeomsState,
    diff_contact_input: array_class.DiffContactInput,
    gjk_info: array_class.GJKInfo,
    i_ga,
    i_gb,
    i_b,
    i_c,
    ref_penetration,
):
    ...


@ti.func
def func_plane_normal(v1, v2, v3):
    """
    Compute the normal of the plane defined by three points. The length of the normal corresponds to the two times the
    area of the triangle.
    """
    d21 = v2 - v1
    d31 = v3 - v1
    normal = d21.cross(d31)
    return normal


@ti.func
def func_project_origin_to_plane(v1, v2, v3, normal):
    """
    Project the origin onto the plane defined by the simplex vertices.

    @ normal: The face normal computed as (v2 - v1) x (v3 - v1). Its length should be guaranteed to be larger than the
    minimum normal norm in [gjk_info], but we do not check it here.
    """
    # Since normal norm is guaranteed to be larger than sqrt(10 * EPS), [nn] is guaranteed to be larger than 10 * EPS.
    v = v1
    nv = normal.dot(v)
    nn = normal.norm_sqr()
    return normal * (nv / nn)


@ti.func
def func_triangle_affine_coords(v1, v2, v3, normal, point):
    """
    Compute the affine coordinates of the point with respect to the triangle.

    @ normal: The face normal computed as (v2 - v1) x (v3 - v1). Its length should be guaranteed to be larger than the
    minimum normal norm in [gjk_info], but we do not check it here.
    @ point: The point on the plane that we want to compute the affine coordinates of.
    """
    # Since normal norm is guaranteed to be larger than sqrt(10 * EPS), [nn] is guaranteed to be larger than 10 * EPS.
    nn = normal.norm_sqr()
    inv_nn = 1.0 / nn

    return gs.ti_vec3(
        (v2 - point).cross(v3 - point).dot(normal) * inv_nn,
        (v3 - point).cross(v1 - point).dot(normal) * inv_nn,
        (v1 - point).cross(v2 - point).dot(normal) * inv_nn,
    )
