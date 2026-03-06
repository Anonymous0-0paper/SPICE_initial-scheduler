from pydantic import BaseModel

from salsa.core.states.util import vector_dim


# --- Microservice Status ---
# Aggregates Workload, Scaling, and SLO metrics for a specific service
class MicroserviceState(BaseModel):
    service_id: str
    app_id: str

    # Workload Metrics {lambda_i, q_i, rt_i}
    request_rate: float  # lambda_i(t)
    queue_length: int  # q_i(t)
    response_time: float  # rt_i(t) in ms

    # Replica States {rho, rho_effective} [cite: 480-481]
    replicas_desired: int  # rho_i,l(t)
    replicas_effective: int  # rho_i,l^effective(t) (Accounts for startup delay)
    replicas_starting: int  # Number of pods currently in the "ContainerCreating" state

    # Action Constraints {t_mig, t_scale} [cite: 698-699]
    seconds_since_last_scale: float
    seconds_since_last_migrate: float
    is_migration_in_progress: bool  # delta_i,l(t)

    def get_dim(self):
        return vector_dim(self)

    def get_as_list(self):
        return [self.request_rate, self.queue_length, self.response_time,
                self.replicas_desired, self.replicas_effective, self.replicas_starting,
                self.seconds_since_last_scale, self.seconds_since_last_migrate, self.is_migration_in_progress]