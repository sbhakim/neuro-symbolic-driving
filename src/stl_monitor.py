# src/stl_monitor.py

from typing import List
import numpy as np


def robustness(distance: float, d_min: float) -> float:
    """
    Simple STL robustness proxy: rho = d - d_min.

    Positive means safe margin, negative means violation.
    """
    return distance - d_min


def compute_rho_min(distances: List[float], d_min: float) -> float:
    """Compute minimum robustness over trajectory."""
    rho_values = [robustness(d, d_min) for d in distances]
    return float(np.min(rho_values))


def compute_satisfaction_rate(distances: List[float], d_min: float) -> float:
    """
    Satisfaction rate: fraction of time steps with rho > 0.

    SR = (1/N) * sum(I[rho_k > 0])
    """
    if len(distances) == 0:
        return 0.0

    satisfied = sum(1 for d in distances if robustness(d, d_min) > 0)
    return satisfied / len(distances)
