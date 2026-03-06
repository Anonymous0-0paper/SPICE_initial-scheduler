from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict

from salsa.utils.typing import cluster_id, microservice_id, application_id

class TierType(Enum):
    EDGE = 1
    FOG = 2
    CLOUD = 3

@dataclass
class Cluster:
    id: cluster_id
    tier_type: TierType

    cost_per_core_hour: float
    cost_per_mem_hour: float # per GiB

    mem_gb: float
    cpu_cores: int

    cpu_core_utilization: float = 0 # between [0, 1]
    mem_utilization: float = 0 # between [0, 1]

    microservices: List[microservice_id] = field(default_factory=list)
    applications: List[application_id] = field(default_factory=list)

    def __post_init__(self):
        self._per_app_microservice_count: Dict[application_id, int]

    def get_resource_cost(self) -> float:
        return (self.cpu_cores * self.cpu_core_utilization * self.cost_per_core_hour + 
                self.mem_gb * self.mem_utilization * self.cost_per_mem_hour)
    
    def get_max_potential_cost(self) -> float:
        return (self.cpu_cores * 1.0 * self.cost_per_core_hour + 
                self.mem_gb * 1.0 * self.cost_per_mem_hour)

    def get_network_latency(self, dest_tier: str) -> float:
        source_tier = self.tier_type.name.lower()
        latencies = {
            frozenset(['cloud', 'edge']): 60.0,
            frozenset(['edge', 'fog']): 15.0,
            frozenset(['cloud', 'fog']): 80.0,
            frozenset(['cloud']): 75.0,
            frozenset(['edge']): 20.0,
            frozenset(['fog']): 5.0
        }

        if source_tier == dest_tier:
            return latencies.get(frozenset([source_tier]), 10.0)

        return latencies.get(frozenset([source_tier, dest_tier]), 50.0)
