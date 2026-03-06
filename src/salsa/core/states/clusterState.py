from pydantic import BaseModel, Field

from salsa.core.states.util import vector_dim


# --- Cluster-Level Resources ---
# Corresponds to {u_{j}^cpu(t), u_{j}^mem(t)}
class ClusterState(BaseModel):
    cluster_id: str
    cpu_utilization: float = Field(..., ge=0.0, le=1.0, description="Normalized CPU usage")
    mem_utilization: float = Field(..., ge=0.0, le=1.0, description="Normalized Memory usage")

    def get_dim(self):
        return vector_dim(self)

    def get_as_list(self):
        return [self.cpu_utilization, self.mem_utilization]
