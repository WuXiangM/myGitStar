import random
import threading
import time as _time


class SimpleThrottle:
    def __init__(self, qps: float):
        self.interval = 1.0 / qps if qps and qps > 0 else 0.0
        self.lock = threading.Lock()
        self.next_allowed = 0.0

    def wait(self) -> None:
        if self.interval <= 0:
            return
        with self.lock:
            now = _time.time()
            if now < self.next_allowed:
                to_sleep = self.next_allowed - now
                _time.sleep(to_sleep + random.uniform(0, 0.1))
                now = _time.time()
            self.next_allowed = now + self.interval
