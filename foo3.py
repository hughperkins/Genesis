import numpy as np

import genesis as gs

gs.init(backend=gs.metal, precision="32")

scene = gs.Scene(
    fem_options=gs.options.FEMOptions(
        use_implicit_solver=True,
    ),
    coupler_options=gs.options.SAPCouplerOptions(),
)
scene.add_entity(
    morph=gs.morphs.Sphere(
        pos=(0.0, 0.0, 0.0),
        radius=0.1,
    ),
    material=gs.materials.FEM.Elastic(
        model="linear_corotated",
    ),
)

scene.build()
print(type(scene.sim.coupler.contact_handlers[1]))
has_contact, _overflow = scene.sim._coupler.update_contact(0)
print("has_contact:", has_contact, "- num contacts:", scene.sim._coupler.contact_handlers[0].n_contact_pairs.to_numpy())
# has_contact: 0 - num contacts: 172