import threading
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Deque

import numpy as np

from salsa.entities.dependencyGraph import DependencyGraph
from salsa.entities.microservice import Microservice
from salsa.utils.datastructures import LimitedDict
from salsa.utils.typing import application_id, microservice_id


@dataclass
class Application:
    id: application_id
    microservices: List[Microservice]
    dependency_graph: DependencyGraph
    entrypoint: str
    slos: Dict[str, float]
    penalty_coefficient: float  # penalty in case of SLO violation
    
    migration_interval: int
    scaling_interval: int
    horizon: int
    
    def __post_init__(self):
        self.sloTracker: SloViolationTracker = SloViolationTracker(self.id, horizon=self.horizon)
        self.lock = threading.Lock()
        self.undeployed_microservices: List[microservice_id] = [ms.id for ms in self.microservices]
        self.is_deployed = False
        self.receives_work = False
    
    def get_receives_work(self):
        with self.lock:
            return self.receives_work

    def set_receives_work(self, val):
        with self.lock:
            self.receives_work = val

class SloViolationTracker:
    def __init__(self, app_id: application_id, horizon: int = 300):
        self.app_id = app_id

        self._lock = threading.Lock()
        self.violation_history: LimitedDict[int, bool] = LimitedDict(limit=3 * horizon)
        self.violation_prediction_history: LimitedDict[int, float] = LimitedDict(limit=horizon + np.log(horizon))
        self.pending_violation_prediction: float = 0.0
        self.pending_violation: bool = False
        self.latest_key = 0

    def report_violation(self) -> None:
        with self._lock:
            self.pending_violation = True

    def flush_violations(self, rnd: int) -> None:
        with self._lock:
            self.violation_history[rnd] = self.pending_violation
            self.pending_violation = False
            self.latest_key = rnd

    def get_latest(self) -> bool:
        with self._lock:
            return self.violation_history[self.latest_key]

    def report_violation_prediction(self, probability: float):
        with self._lock:
            self.pending_violation_prediction = probability

    def flush_violation_prediction(self, rnd: int):
        with self._lock:
            self.violation_prediction_history[rnd] = self.pending_violation_prediction
            self.pending_violation_prediction = 0.0

    def get_violation_prediction(self, rnd: int):
        with self._lock:
            return self.violation_prediction_history.get(rnd, 0.0)

    def clear_history(self):
        with self._lock:
            self.violation_history.clear()
            self.violation_prediction_history.clear()
    
