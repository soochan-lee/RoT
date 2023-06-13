import random
import time


class Reservoir(list):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.total = 0
        self._evict = None

    def reserve(self) -> bool:
        self.total += 1
        if len(self) < self.size:
            return True
        self._evict = random.randrange(0, self.total)
        return self._evict < self.size

    def add(self, x) -> None:
        if len(self) < self.size:
            self.append(x)
        else:
            self[self._evict] = x
            self._evict = None

    def inflow(self, x) -> None:
        if self.reserve():
            self.add(x)


class Timer:
    def __init__(self, text):
        self.text = text

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        elapsed_time = time.perf_counter() - self._start
        print(self.text.format(elapsed_time))
