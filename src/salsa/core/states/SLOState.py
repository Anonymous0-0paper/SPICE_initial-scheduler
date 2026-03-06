from typing import List

from pydantic import BaseModel

from salsa.core.states.util import vector_dim


# --- Application-Level SLOs ---
# Corresponds to {l_i, theta_i, alpha_i} and Distance-to-Thresholds
class SLOState(BaseModel):
    app_id: str

    current_latency: float  # l_i(t)
    current_throughput: float  # theta_i(t)
    current_availability: float  # alpha_i(t)

    is_violating: bool  # v_i(t)
    violation_history_window: List[bool]  # v_i(t-tau:t) - Last N steps

    # Predictive Features [cite: 1061]
    predicted_violation_prob: float  # v_hat_i(t + Delta) from the LSTM predictor

    dist_latency: float  # d_i^lat(t)
    dist_throughput: float  # d_i^thr(t)

    def get_dim(self):
        return vector_dim(self)

    def get_as_list(self) -> List[float]:
        violating_float = 1.0 if self.is_violating else 0.0

        if self.violation_history_window:
            violation_rate = sum(self.violation_history_window) / len(self.violation_history_window)
        else:
            violation_rate = 0.0

        return [
            self.current_latency,
            self.current_throughput,
            self.current_availability,
            violating_float,
            violation_rate,
            self.predicted_violation_prob,
            self.dist_latency,
            self.dist_throughput
        ]
