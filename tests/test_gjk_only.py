"""Test GJK sphere-capsule collision in isolation (no analytical scene)."""

import os
import tempfile
import xml.etree.ElementTree as ET
import genesis as gs

def create_sphere_mjcf(name, pos, radius):
    mjcf = ET.Element("mujoco", model=name)
    ET.SubElement(mjcf, "compiler", angle="degree")
    ET.SubElement(mjcf, "option", timestep="0.01")
    worldbody = ET.SubElement(mjcf, "worldbody")
    body = ET.SubElement(worldbody, "body", name=name, pos=f"{pos[0]} {pos[1]} {pos[2]}")
    ET.SubElement(body, "geom", type="sphere", size=f"{radius}")
    ET.SubElement(body, "joint", name=f"{name}_joint", type="free")
    return mjcf

def create_capsule_mjcf(name, pos, euler, radius, half_length):
    mjcf = ET.Element("mujoco", model=name)
    ET.SubElement(mjcf, "compiler", angle="degree")
    ET.SubElement(mjcf, "option", timestep="0.01")
    worldbody = ET.SubElement(mjcf, "worldbody")
    body = ET.SubElement(
        worldbody, "body", name=name,
        pos=f"{pos[0]} {pos[1]} {pos[2]}",
        euler=f"{euler[0]} {euler[1]} {euler[2]}"
    )
    ET.SubElement(body, "geom", type="capsule", size=f"{radius} {half_length}")
    ET.SubElement(body, "joint", name=f"{name}_joint", type="free")
    return mjcf

def test_gjk_sphere_capsule_isolated():
    """Test GJK collision detection for sphere-capsule WITHOUT analytical scene."""
    
    # Create ONLY the GJK scene
    scene_gjk = gs.Scene(
        show_viewer=False,
        rigid_options=gs.options.RigidOptions(
            dt=0.01,
            gravity=(0, 0, 0),
            use_gjk_collision=True,
        ),
    )
    
    sphere_pos = (0.15, 0, 0.3)
    capsule_pos = (0, 0, 0)
    capsule_euler = (0, 0, 0)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        sphere_mjcf = create_sphere_mjcf("sphere", sphere_pos, 0.1)
        sphere_path = os.path.join(tmpdir, "sphere.xml")
        ET.ElementTree(sphere_mjcf).write(sphere_path)
        scene_gjk.add_entity(gs.morphs.MJCF(file=sphere_path))

        capsule_mjcf = create_capsule_mjcf("capsule", capsule_pos, capsule_euler, 0.1, 0.25)
        capsule_path = os.path.join(tmpdir, "capsule.xml")
        ET.ElementTree(capsule_mjcf).write(capsule_path)
        scene_gjk.add_entity(gs.morphs.MJCF(file=capsule_path))

        scene_gjk.build()

    scene_gjk.step()
    
    contacts = scene_gjk.rigid_solver.collider.get_contacts(as_tensor=False, to_torch=False)
    has_collision = contacts is not None and len(contacts.get("geom_a", [])) > 0
    
    print(f"\n=== GJK-ONLY TEST ===")
    print(f"Sphere pos: {sphere_pos}, radius: 0.1")
    print(f"Capsule pos: {capsule_pos}, euler: {capsule_euler}, radius: 0.1, half_length: 0.25")
    print(f"GJK collision detected: {has_collision}")
    
    if has_collision:
        print(f"  Penetration: {contacts['penetration'][0]}")
        print(f"  Normal: {contacts['normal'][0]}")
        print(f"  Position: {contacts['position'][0]}")
        print("SUCCESS: GJK detected the collision")
    else:
        print("FAIL: GJK did not detect collision")
        assert False, "GJK should detect collision but didn't"
