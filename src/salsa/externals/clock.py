import time
import threading
from typing import Dict

class EventClock:
    def __init__(self):
        self._clocks: Dict[str, float] = {}
        self._lock = threading.Lock()

    def touch(self, event_id: str) -> None:
        with self._lock:
            self._clocks[event_id] = time.monotonic()

    def get_time_since(self, event_id: str, default: float = -1.0) -> float:
        with self._lock:
            start_time = self._clocks.get(event_id)
            if start_time is None:
                return default
            return time.monotonic() - start_time

    def clear(self, event_id: str) -> None:
        with self._lock:
            self._clocks.pop(event_id, None)

    def get_all_durations(self) -> Dict[str, float]:
        now = time.monotonic()
        with self._lock:
            return {k: now - v for k, v in self._clocks.items()}