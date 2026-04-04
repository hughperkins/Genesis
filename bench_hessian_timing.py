"""Measure per-substep solver kernel time with CUDA graph ON via CUDA events."""
import sys
import time
import torch
import quadrants as qd

sys.path.insert(0, "tests")
import importlib, types
tests_pkg = types.ModuleType("tests")
tests_pkg.__path__ = ["tests"]
sys.modules["tests"] = tests_pkg
import genesis as gs
gs.init()
from tests.test_rigid_benchmarks import make_dex_hand

N_ENVS = 4096
WARMUP_STEPS = 30
MEASURE_STEPS = 20

_, step_fn, meta = make_dex_hand(N_ENVS)
qd.sync()

for i in range(WARMUP_STEPS):
    step_fn()
qd.sync()

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

times = []
for i in range(MEASURE_STEPS):
    start_event.record()
    step_fn()
    end_event.record()
    torch.cuda.synchronize()
    times.append(start_event.elapsed_time(end_event))

avg_ms = sum(times) / len(times)
fps = N_ENVS / (avg_ms / 1000)
print(f"avg step: {avg_ms:.3f} ms, FPS: {fps:.0f}")
print(f"per-step times (ms): {', '.join(f'{t:.3f}' for t in times)}")
