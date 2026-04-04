"""3-way benchmark: clean main vs branch-always-full vs branch-adaptive."""
import subprocess, sys, os, time

SCRIPT = """
import sys, types, time
sys.path.insert(0, "tests")
tests_pkg = types.ModuleType("tests")
tests_pkg.__path__ = ["tests"]
sys.modules["tests"] = tests_pkg
import genesis as gs; gs.init()
from tests.test_rigid_benchmarks import make_dex_hand
import quadrants as qd

N_ENVS = 4096
WARMUP = 100
MEASURE = 200

scene, step_fn, _ = make_dex_hand(N_ENVS)
qd.sync()
for i in range(WARMUP):
    step_fn()
qd.sync()
t0 = time.perf_counter()
for i in range(MEASURE):
    step_fn()
qd.sync()
dt = time.perf_counter() - t0
fps = MEASURE * N_ENVS / dt
print(f"RESULT: {fps:.0f} FPS ({dt*1000/MEASURE:.2f} ms/step)")
"""

def run_bench(label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}", flush=True)
    r = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        env={**os.environ, "GS_ENABLE_NDARRAY": "0"},
        capture_output=True, text=True, timeout=600,
    )
    for line in r.stdout.splitlines():
        if "RESULT" in line:
            print(line)
            return
    print("FAILED")
    print(r.stderr[-500:] if r.stderr else "no stderr")

run_bench("CURRENT (adaptive H patching)")
