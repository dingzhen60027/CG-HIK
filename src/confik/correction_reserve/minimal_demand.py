"""Causal, untruncated empirical prediction demand; no probability guarantee."""
from collections import deque
import math
import numpy as np

from ..geometry import pose_error
from .geometry import predict_target


def empirical_quantile(values, probability):
    """Inverse empirical CDF (an observed order statistic, no interpolation)."""
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or not len(a) or not np.isfinite(a).all():
        raise ValueError('finite nonempty scalar observations required')
    if not 0 < probability <= 1:
        raise ValueError('quantile probability outside (0, 1]')
    index = math.ceil(probability * len(a)) - 1
    return float(np.partition(a, index)[index])


class PredictionDemand:
    def __init__(self, scale, minimum, initial, *, fixed=False, window=20):
        if window != 20 or not np.isfinite([minimum, initial]).all() or min(minimum, initial) < 0:
            raise ValueError('frozen window 20 and finite nonnegative demand required')
        self.scale = np.asarray(scale, float)
        self.minimum, self.initial, self.fixed = float(minimum), float(initial), bool(fixed)
        self.window = window
        self.reset()

    def reset(self):
        self.last_target = None
        self.forecast = None
        self.errors = deque(maxlen=self.window)

    def observe(self, target):
        eta = None
        if self.forecast is not None:
            eta = float(np.linalg.norm(pose_error(target, self.forecast) / self.scale))
            if not np.isfinite(eta):
                raise ValueError('nonfinite observed prediction error')
            self.errors.append(eta)
        predicted = predict_target(target, self.last_target)
        self.last_target, self.forecast = target, predicted
        startup = len(self.errors) < self.window
        quantile = self.initial if self.fixed or startup else empirical_quantile(self.errors, .95)
        demand = max(self.minimum, quantile)  # deliberately no upper clipping
        return predicted, dict(demand=demand, observed_eta=eta, demand_history_count=len(self.errors),
            demand_startup=bool(startup), demand_fixed=self.fixed, demand_quantile=quantile)
