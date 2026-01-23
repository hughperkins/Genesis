"""Unit tests for func_hessian_direct_tiled."""
import numpy as np
import pytest
import gstaichi as ti

import genesis as gs
from genesis.utils import array_class
from genesis.engine.solvers.rigid.constraint.solver import (
    func_hessian_direct_tiled,
    func_hessian_direct_batch,
)

from .utils import assert_allclose


def create_test_constraint_state(n_constraints, n_dofs, batch_size):
    """Create a minimal ConstraintState for testing.
    
    Note: Genesis must be initialized (gs.init()) before calling this function.
    """
    # Create the data structures
    constraint_state = array_class.StructConstraintState(
        n_constraints=array_class.V(dtype=gs.ti_int, shape=(batch_size,)),
        ti_n_equalities=array_class.V(dtype=gs.ti_int, shape=(batch_size,)),
        n_constraints_equality=array_class.V(dtype=gs.ti_int, shape=(batch_size,)),
        n_constraints_frictionloss=array_class.V(dtype=gs.ti_int, shape=(batch_size,)),
        is_warmstart=array_class.V(dtype=gs.ti_bool, shape=(batch_size,)),
        improved=array_class.V(dtype=gs.ti_bool, shape=(batch_size,)),
        cost_ws=array_class.V(dtype=gs.ti_float, shape=(batch_size,)),
        gauss=array_class.V(dtype=gs.ti_float, shape=(batch_size,)),
        cost=array_class.V(dtype=gs.ti_float, shape=(batch_size,)),
        prev_cost=array_class.V(dtype=gs.ti_float, shape=(batch_size,)),
        gtol=array_class.V(dtype=gs.ti_float, shape=(batch_size,)),
        ls_it=array_class.V(dtype=gs.ti_int, shape=(batch_size,)),
        ls_result=array_class.V(dtype=gs.ti_int, shape=(batch_size,)),
        cg_beta=array_class.V(dtype=gs.ti_float, shape=(batch_size,)),
        cg_pg_dot_pMg=array_class.V(dtype=gs.ti_float, shape=(batch_size,)),
        quad_gauss=array_class.V(dtype=gs.ti_float, shape=(3, batch_size)),
        candidates=array_class.V(dtype=gs.ti_float, shape=(12, batch_size)),
        Ma=array_class.V(dtype=gs.ti_float, shape=(n_dofs, batch_size)),
        Ma_ws=array_class.V(dtype=gs.ti_float, shape=(n_dofs, batch_size)),
        grad=array_class.V(dtype=gs.ti_float, shape=(n_dofs, batch_size)),
        Mgrad=array_class.V(dtype=gs.ti_float, shape=(n_dofs, batch_size)),
        search=array_class.V(dtype=gs.ti_float, shape=(n_dofs, batch_size)),
        qfrc_constraint=array_class.V(dtype=gs.ti_float, shape=(n_dofs, batch_size)),
        qacc=array_class.V(dtype=gs.ti_float, shape=(n_dofs, batch_size)),
        qacc_ws=array_class.V(dtype=gs.ti_float, shape=(n_dofs, batch_size)),
        qacc_prev=array_class.V(dtype=gs.ti_float, shape=(n_dofs, batch_size)),
        mv=array_class.V(dtype=gs.ti_float, shape=(n_dofs, batch_size)),
        cg_prev_grad=array_class.V(dtype=gs.ti_float, shape=(n_dofs, batch_size)),
        cg_prev_Mgrad=array_class.V(dtype=gs.ti_float, shape=(n_dofs, batch_size)),
        nt_vec=array_class.V(dtype=gs.ti_float, shape=(n_dofs, batch_size)),
        nt_H=array_class.V(dtype=gs.ti_float, shape=(batch_size, n_dofs, n_dofs)),
        efc_b=array_class.V(dtype=gs.ti_float, shape=(n_constraints, batch_size)),
        efc_AR=array_class.V(dtype=gs.ti_float, shape=(n_constraints, n_constraints, batch_size)),
        active=array_class.V(dtype=gs.ti_bool, shape=(n_constraints, batch_size)),
        prev_active=array_class.V(dtype=gs.ti_bool, shape=(n_constraints, batch_size)),
        diag=array_class.V(dtype=gs.ti_float, shape=(n_constraints, batch_size)),
        aref=array_class.V(dtype=gs.ti_float, shape=(n_constraints, batch_size)),
        Jaref=array_class.V(dtype=gs.ti_float, shape=(n_constraints, batch_size)),
        efc_frictionloss=array_class.V(dtype=gs.ti_float, shape=(n_constraints, batch_size)),
        efc_force=array_class.V(dtype=gs.ti_float, shape=(n_constraints, batch_size)),
        efc_D=array_class.V(dtype=gs.ti_float, shape=(n_constraints, batch_size)),
        jv=array_class.V(dtype=gs.ti_float, shape=(n_constraints, batch_size)),
        quad=array_class.V(dtype=gs.ti_float, shape=(n_constraints, 3, batch_size)),
        jac=array_class.V(dtype=gs.ti_float, shape=(n_constraints, n_dofs, batch_size)),
        jac_relevant_dofs=array_class.V(dtype=gs.ti_int, shape=()),  # Not used in dense mode
        jac_n_relevant_dofs=array_class.V(dtype=gs.ti_int, shape=()),  # Not used in dense mode
        # Backward gradients (empty for this test)
        dL_dqacc=array_class.V(dtype=gs.ti_float, shape=()),
        dL_dM=array_class.V(dtype=gs.ti_float, shape=()),
        dL_djac=array_class.V(dtype=gs.ti_float, shape=()),
        dL_daref=array_class.V(dtype=gs.ti_float, shape=()),
        dL_defc_D=array_class.V(dtype=gs.ti_float, shape=()),
        dL_dforce=array_class.V(dtype=gs.ti_float, shape=()),
        bw_u=array_class.V(dtype=gs.ti_float, shape=()),
        bw_r=array_class.V(dtype=gs.ti_float, shape=()),
        bw_p=array_class.V(dtype=gs.ti_float, shape=()),
        bw_Ap=array_class.V(dtype=gs.ti_float, shape=()),
        bw_Ju=array_class.V(dtype=gs.ti_float, shape=()),
        bw_y=array_class.V(dtype=gs.ti_float, shape=()),
        bw_w=array_class.V(dtype=gs.ti_float, shape=()),
        # Timers
        timers=array_class.V(dtype=ti.i64 if gs.backend != gs.metal else ti.i32, shape=(10, batch_size)),
    )
    
    return constraint_state


def create_test_rigid_global_info(n_dofs, batch_size):
    """Create a minimal RigidGlobalInfo for testing.
    
    Note: Genesis must be initialized (gs.init()) before calling this function.
    """
    
    # Create minimal placeholder fields
    rigid_global_info = array_class.StructRigidGlobalInfo(
        n_awake_dofs=array_class.V(dtype=gs.ti_int, shape=(batch_size,)),
        awake_dofs=array_class.V(dtype=gs.ti_int, shape=(n_dofs, batch_size)),
        n_awake_entities=array_class.V(dtype=gs.ti_int, shape=(batch_size,)),
        awake_entities=array_class.V(dtype=gs.ti_int, shape=(1, batch_size)),
        n_awake_links=array_class.V(dtype=gs.ti_int, shape=(batch_size,)),
        awake_links=array_class.V(dtype=gs.ti_int, shape=(1, batch_size)),
        qpos0=array_class.V(dtype=gs.ti_float, shape=(n_dofs, batch_size)),
        qpos=array_class.V(dtype=gs.ti_float, shape=(n_dofs, batch_size)),
        qpos_next=array_class.V(dtype=gs.ti_float, shape=(n_dofs, batch_size)),
        links_T=array_class.V(dtype=gs.ti_mat4, shape=(1, batch_size)),
        envs_offset=array_class.V(dtype=gs.ti_vec3, shape=(batch_size,)),
        geoms_init_AABB=array_class.V(dtype=gs.ti_float, shape=(6, 1, batch_size)),
        mass_mat=array_class.V(dtype=gs.ti_float, shape=(n_dofs, n_dofs, batch_size)),
        mass_mat_L=array_class.V(dtype=gs.ti_float, shape=(n_dofs, n_dofs, batch_size)),
        mass_mat_L_bw=array_class.V(dtype=gs.ti_float, shape=()),
        mass_mat_D_inv=array_class.V(dtype=gs.ti_float, shape=(n_dofs, batch_size)),
        mass_mat_mask=array_class.V(dtype=gs.ti_float, shape=(n_dofs, n_dofs, batch_size)),
        meaninertia=array_class.V(dtype=gs.ti_float, shape=(batch_size,)),
        mass_parent_mask=array_class.V(dtype=gs.ti_float, shape=(n_dofs, n_dofs, batch_size)),
        EPS=array_class.V(dtype=gs.ti_float, shape=()),
        dL_dqpos=array_class.V(dtype=gs.ti_float, shape=()),
        dL_dlinks_T=array_class.V(dtype=gs.ti_mat4, shape=()),
        dL_dlinks_T_aux=array_class.V(dtype=gs.ti_mat4, shape=()),
        dL_dmass_mat=array_class.V(dtype=gs.ti_float, shape=()),
        dL_dC=array_class.V(dtype=gs.ti_float, shape=()),
        dL_dC_aux=array_class.V(dtype=gs.ti_float, shape=()),
    )
    
    # Set EPS
    rigid_global_info.EPS.fill(1e-8)
    
    return rigid_global_info


def create_test_entities_info(n_dofs, batch_size):
    """Create a minimal EntitiesInfo for testing.
    
    Note: Genesis must be initialized (gs.init()) before calling this function.
    """
    
    entities_info = array_class.StructEntitiesInfo(
        batch_idx=array_class.V(dtype=gs.ti_int, shape=(1,)),
        n_links=array_class.V(dtype=gs.ti_int, shape=(1,)),
        n_qs=array_class.V(dtype=gs.ti_int, shape=(1,)),
        n_dofs=array_class.V(dtype=gs.ti_int, shape=(1,)),
        dof_start=array_class.V(dtype=gs.ti_int, shape=(1,)),
        dof_end=array_class.V(dtype=gs.ti_int, shape=(1,)),
        q_start=array_class.V(dtype=gs.ti_int, shape=(1,)),
        q_end=array_class.V(dtype=gs.ti_int, shape=(1,)),
        link_start=array_class.V(dtype=gs.ti_int, shape=(1,)),
        geom_start=array_class.V(dtype=gs.ti_int, shape=(1,)),
        n_geoms=array_class.V(dtype=gs.ti_int, shape=(1,)),
        invweight=array_class.V(dtype=gs.ti_float, shape=(n_dofs,)),
        link_parent=array_class.V(dtype=gs.ti_int, shape=(1,)),
        link_entity=array_class.V(dtype=gs.ti_int, shape=(1,)),
        geom_entity=array_class.V(dtype=gs.ti_int, shape=(1,)),
        geom_link=array_class.V(dtype=gs.ti_int, shape=(1,)),
        geom_collision_idx=array_class.V(dtype=gs.ti_int, shape=(1,)),
    )
    
    # Set up one entity that spans all DOFs
    entities_info.n_links.fill(1)
    entities_info.n_dofs.fill(n_dofs)
    entities_info.dof_start.fill(0)
    entities_info.dof_end.fill(n_dofs)
    
    return entities_info


@ti.kernel
def initialize_test_data(
    constraint_state: ti.template(),
    rigid_global_info: ti.template(),
    n_constraints: int,
    n_dofs: int,
    batch_size: int,
):
    """Initialize test data with random but valid values."""
    for i_b in range(batch_size):
        # Set number of constraints and mark as improved
        constraint_state.n_constraints[i_b] = n_constraints
        constraint_state.improved[i_b] = True
        
        # Initialize Jacobian with random values
        for i_c in range(n_constraints):
            constraint_state.active[i_c, i_b] = True
            constraint_state.efc_D[i_c, i_b] = 1.0 + 0.1 * float(i_c)  # Positive values
            
            for i_d in range(n_dofs):
                # Create some structure: most entries zero, some non-zero
                if (i_c + i_d) % 3 == 0:
                    constraint_state.jac[i_c, i_d, i_b] = 0.1 * float(i_c + i_d + 1)
                else:
                    constraint_state.jac[i_c, i_d, i_b] = 0.0
        
        # Initialize mass matrix (make it symmetric positive definite)
        for i_d1 in range(n_dofs):
            for i_d2 in range(i_d1 + 1):
                if i_d1 == i_d2:
                    # Diagonal entries - positive
                    rigid_global_info.mass_mat[i_d1, i_d2, i_b] = 1.0 + 0.1 * float(i_d1)
                else:
                    # Off-diagonal entries - small
                    rigid_global_info.mass_mat[i_d1, i_d2, i_b] = 0.01 * float(i_d1 + i_d2)
                    rigid_global_info.mass_mat[i_d2, i_d1, i_b] = 0.01 * float(i_d1 + i_d2)


@pytest.mark.required
@pytest.mark.parametrize("backend", [gs.gpu])
@pytest.mark.parametrize("problem_size", [
    # (n_constraints, n_dofs, batch_size)
    (8, 16, 2),    # Small case
    (16, 32, 4),   # Medium case
    (32, 64, 2),   # Large case - fits in single block
    (48, 80, 2),   # Requires tiling
])
def test_func_hessian_direct_tiled(backend, problem_size):
    """Test that func_hessian_direct_tiled produces the same results as func_hessian_direct_batch."""
    n_constraints, n_dofs, batch_size = problem_size
    
    # Create test data structures
    constraint_state_tiled = create_test_constraint_state(n_constraints, n_dofs, batch_size)
    constraint_state_batch = create_test_constraint_state(n_constraints, n_dofs, batch_size)
    rigid_global_info = create_test_rigid_global_info(n_dofs, batch_size)
    entities_info = create_test_entities_info(n_dofs, batch_size)
    
    # Initialize with the same test data
    initialize_test_data(constraint_state_tiled, rigid_global_info, n_constraints, n_dofs, batch_size)
    initialize_test_data(constraint_state_batch, rigid_global_info, n_constraints, n_dofs, batch_size)
    
    # Create a simple config for the batch version
    @ti.dataclass
    class TestConfig:
        sparse_solve: bool = False
        backend: int = gs.gpu
        para_level: int = gs.PARA_LEVEL.ALL
    
    static_config = TestConfig()
    
    # Run the tiled version (GPU-optimized)
    @ti.kernel
    def run_tiled():
        func_hessian_direct_tiled(constraint_state_tiled, rigid_global_info)
    
    run_tiled()
    
    # Run the batch version for each environment
    @ti.kernel
    def run_batch():
        for i_b in range(batch_size):
            func_hessian_direct_batch(
                i_b,
                entities_info,
                constraint_state_batch,
                rigid_global_info,
                static_config,
            )
    
    run_batch()
    
    # Extract results and compare
    H_tiled = constraint_state_tiled.nt_H.to_numpy()
    H_batch = constraint_state_batch.nt_H.to_numpy()
    
    # Only compare lower triangular part (upper part is undefined)
    for i_b in range(batch_size):
        for i_d1 in range(n_dofs):
            for i_d2 in range(i_d1 + 1):  # Lower triangular only
                tiled_val = H_tiled[i_b, i_d1, i_d2]
                batch_val = H_batch[i_b, i_d1, i_d2]
                
                # Use relative tolerance since values can be large
                assert_allclose(
                    tiled_val,
                    batch_val,
                    rtol=1e-5,
                    atol=1e-8,
                    err_msg=f"Mismatch at batch {i_b}, position ({i_d1}, {i_d2}): "
                            f"tiled={tiled_val}, batch={batch_val}"
                )


@pytest.mark.required  
@pytest.mark.parametrize("backend", [gs.gpu])
def test_func_hessian_direct_tiled_edge_cases(backend):
    """Test edge cases like zero constraints or all inactive constraints."""
    
    n_constraints, n_dofs, batch_size = 16, 32, 2
    
    # Test with zero constraints
    constraint_state = create_test_constraint_state(n_constraints, n_dofs, batch_size)
    rigid_global_info = create_test_rigid_global_info(n_dofs, batch_size)
    
    # Set n_constraints to 0 for first batch
    constraint_state.n_constraints.from_numpy(np.array([0, 8], dtype=np.int32))
    constraint_state.improved.from_numpy(np.array([True, True], dtype=bool))
    
    # Initialize second batch normally
    @ti.kernel
    def init_second_batch():
        for i_c in range(8):
            constraint_state.active[i_c, 1] = True
            constraint_state.efc_D[i_c, 1] = 1.0
            for i_d in range(n_dofs):
                constraint_state.jac[i_c, i_d, 1] = 0.1
    
    init_second_batch()
    
    # This should not crash
    @ti.kernel
    def run_tiled():
        func_hessian_direct_tiled(constraint_state, rigid_global_info)
    
    run_tiled()
    
    # First batch should have skipped computation (due to n_constraints=0)
    # Second batch should have valid results
    H = constraint_state.nt_H.to_numpy()
    
    # Check that second batch has non-zero values
    assert np.any(H[1] != 0.0), "Second batch should have non-zero Hessian values"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

