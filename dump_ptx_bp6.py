"""Dump PTX for box_pyramid_6 monolith kernel with current qd version."""
import os, sys
os.environ["GS_ENABLE_NDARRAY"] = "0"
os.environ["TI_DUMP_IR"] = "1"

sys.path.insert(0, 'tests')
import types
tests_pkg = types.ModuleType('tests'); tests_pkg.__path__ = ['tests']; sys.modules['tests'] = tests_pkg

import genesis as gs
gs.vis.visualizer.Visualizer.build = lambda self: None
gs.init()

import quadrants as qd
print(f"Quadrants version: {qd.__version__}")

from tests.test_rigid_benchmarks import make_box_pyramid
scene, step_fn, _ = make_box_pyramid(4096, n_cubes=6)
qd.sync()

# Run one step to trigger compilation and PTX dump
step_fn()
qd.sync()
print("Done - check /tmp/qd_dump/ or similar for PTX output")
