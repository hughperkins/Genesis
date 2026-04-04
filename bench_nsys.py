"""Run dex_hand for nsys profiling. Use with: nsys profile -o report python bench_nsys.py"""
import sys, types
sys.path.insert(0, "tests")
tests_pkg = types.ModuleType("tests")
tests_pkg.__path__ = ["tests"]
sys.modules["tests"] = tests_pkg
import genesis as gs
gs.init()
from tests.test_rigid_benchmarks import make_dex_hand
import torch
import quadrants as qd

N_ENVS = 4096
WARMUP = 30
MEASURE = 5

_, step_fn, _ = make_dex_hand(N_ENVS)
qd.sync()

for i in range(WARMUP):
    step_fn()
qd.sync()

torch.cuda.cudart().cudaProfilerStart()
for i in range(MEASURE):
    step_fn()
qd.sync()
torch.cuda.cudart().cudaProfilerStop()
print(f"Done: {MEASURE} steps profiled")
