"""Single-flight control for the one free web instance."""

from __future__ import annotations

import threading


class Runner:
    """One video at a time. A second upload is refused instead of doubling RAM."""

    def __init__(self):
        self.busy = False
        self.stop = False
        self._lock = threading.Lock()

    def try_start(self) -> bool:
        with self._lock:
            if self.busy:
                return False
            self.busy = True
            self.stop = False
            return True

    def finish(self) -> None:
        with self._lock:
            self.busy = False
            self.stop = False

    def request_stop(self) -> None:
        self.stop = True

    def stopped(self) -> bool:
        return self.stop
