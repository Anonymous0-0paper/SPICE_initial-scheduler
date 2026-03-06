import threading
from dataclasses import dataclass
from salsa.utils.typing import microservice_id, application_id


@dataclass
class Microservice:
    id: microservice_id
    app_id: application_id
    cpu_core_demands: float
    mem_demands_bytes: float

    max_tolerated_replicas: int = 1
    desired_replicas: int = 1
    current_replicas: int = 0

    is_migration_in_progress: bool = False
    migration_cost: float = 1
    def __post_init__(self):
        self._lock = threading.Lock()
        self.id += f"_{self.app_id}"

    @property
    def is_migrating(self) -> bool:
        with self._lock:
            return self.is_migration_in_progress
