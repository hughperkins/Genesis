import time
import gstaichi as ti


class ColliderDiagTimings:
    def __init__(self) -> None:
        self.last_print = time.time()
        self.count = 0
        self.cum_time = 0

    def before_call(self) -> None:
        ti.sync()
        self.before = time.time()

    def after_call(self) -> None:
        ti.sync()
        elapsed = time.time() - self.before
        self.cum_time += elapsed
        self.count += 1

        if time.time() - self.last_print >= 3.0:
            it_time_us = self.cum_time / self.count * 1e6
            print(f"It time {it_time_us:.1f}us")
            self.count = 0
            self.cum_time = 0
            self.last_print = time.time()
