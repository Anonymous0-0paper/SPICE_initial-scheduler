from collections import deque
from typing import Dict, Any

import numpy as np
from scipy.stats import norm

import pandas as pd

from salsa.sloViolationPredictor.base_predictor import BasePredictor


class StatisticalSLOPredictor(BasePredictor):
    def __init__(self, horizon: int, is_min_metric: bool, threshold, alpha=0.3, beta=0.1):
        self.horizon = horizon      # Number of timesteps to predict ahead
        self.threshold = threshold

        # is_min_metric is True for metrics that are optimized using a minimization objective (e.g., Latency)
        self.distribution_fn = norm.sf if is_min_metric else norm.cdf

        self.level = None
        self.trend = None

        self.alpha = alpha
        self.beta = beta

        self.errors = deque(maxlen=horizon * 20)

        self.last_value = 0.0

    def predict(self, df: pd.DataFrame, context: Dict[str, Any]) -> float:
        """
        Returns the probability (0.0 to 1.0) of an SLO violation 'horizon' steps from now.
        """
        # Ingest new data point
        current_value = self.add_df_to_history(val=context["app_val"])

        # Handle cold start
        if self.level is None:
            return 0.0

        # Forecast the value 'horizon' steps ahead
        # Formula: y_hat = Level + (Horizon * Trend)
        expected_val = self.level + (self.horizon * self.trend)

        # Estimate Uncertainty (Sigma)
        # We use the standard deviation of recent forecast errors.
        if len(self.errors) > 2:
            sigma = np.std(self.errors) + 1e-6
        else:
            sigma = abs(current_value) * 0.05 + 1e-6

        # Calculate Probability of Violation
        probability = self.distribution_fn(self.threshold, loc=expected_val, scale=sigma)

        return float(probability)

    def add_df_to_history(self, val: float) -> float:
        """
        Updates the internal Exponential Smoothing state (Level & Trend).
        """
        # Initialize if first point
        if self.level is None:
            self.level = val
            self.trend = 0.0
            self.last_value = val
            return val

        last_level = self.level
        last_trend = self.trend

        # Update Level: weighted average of (current observation) and (predicted level from last step)
        # Level_t = alpha * Value_t + (1 - alpha) * (Level_{t-1} + Trend_{t-1})
        self.level = self.alpha * val + (1 - self.alpha) * (last_level + last_trend)

        # Update Trend: weighted average of (change in levels) and (previous trend)
        # Trend_t = beta * (Level_t - Level_{t-1}) + (1 - beta) * Trend_{t-1}
        self.trend = self.beta * (self.level - last_level) + (1 - self.beta) * last_trend

        # Track Error for Sigma calculation
        one_step_prediction = last_level + last_trend
        error = val - one_step_prediction
        self.errors.append(error)

        self.last_value = val
        return val

    def reset(self):
        """
        Clears all internal state for a fresh start.
        Call this when the DRL environment resets.
        """
        self.level = None
        self.trend = None
        self.errors.clear()
        self.last_value = 0.0
