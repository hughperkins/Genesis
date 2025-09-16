import numpy as np

import genesis as gs

gs.init(backend=gs.metal, precision="32", debug=True)

scene = gs.Scene(
    fem_options=gs.options.FEMOptions(
        use_implicit_solver=True,
    ),
    coupler_options=gs.options.SAPCouplerOptions(),
)
scene.add_entity(
    morph=gs.morphs.Sphere(
        pos=(0.5, -0.2, 0.5),
        radius=0.1,
    ),
    material=gs.materials.FEM.Elastic(
        model="linear_corotated",
    ),
)

# scene.build()
scene.n_envs = scene.sim.n_envs = 0
scene._B = scene.sim._B = 1
scene.envs_offset = np.array(((0.0, 0.0, 0.0),), dtype=gs.np_float)
scene.sim._para_level = gs.PARA_LEVEL.PARTIAL
scene.sim.fem_solver.build()
scene.sim._coupler.build()
scene.sim._coupler.update_contact(0)
