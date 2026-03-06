from pydantic import BaseModel

from salsa.core.states.util import vector_dim


# --- Neighbor Visibility ---
# Corresponds to {u_j'^agg, SLO_j'^agg} used for attention mechanism
class NeighborClusterState(BaseModel):
    cluster_id: str

    cpu_utilization: float  # Aggregated resource usage
    mem_utilization: float
    global_slo_risk_score: float  # Aggregated SLO violation risk
    network_latency_to_neighbor: float  # L_j,j'

    def get_dim(self):
        return vector_dim(self)

    def get_as_list(self):
        return [self.cpu_utilization, self.mem_utilization,
                self.global_slo_risk_score, self.network_latency_to_neighbor]
