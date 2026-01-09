# src/authority_allocator.py

from typing import Optional
import numpy as np
from .llm_interface import AuthorityIntent


def project_invariants(alpha_prop: float, alpha_k: float,
                       emergency: bool, dt: float, gamma: float) -> float:
    """
    Project proposal to invariant set (rate limit + emergency monotonicity).

    I_k = {alpha: |alpha - alpha_k| <= gamma*dt} ∩ [alpha_k, 1] if emergency
                                                  ∩ [0, 1] otherwise
    """
    lo = alpha_k - gamma * dt
    hi = alpha_k + gamma * dt

    if emergency:
        lo = max(lo, alpha_k)  # monotonicity: alpha must not decrease

    lo = max(lo, 0.0)
    hi = min(hi, 1.0)

    return float(np.clip(alpha_prop, lo, hi))


class ClassicalAllocator:
    """Baseline 1: Rule-based authority allocation with risk thresholds."""

    def __init__(self, tau_up: float = 2.5, tau_down: float = 5.0,
                 d_down: float = 25.0, delta_alpha_up: float = 0.05,
                 delta_alpha_down: float = 0.02, alpha_min: float = 0.15,
                 sigma_alpha: float = 0.03,
                 rng: Optional[np.random.Generator] = None):
        """
        Args:
            tau_up: TTC threshold to increase authority [s]
            tau_down: TTC threshold to decrease authority [s]
            d_down: Distance threshold for decrease [m]
            delta_alpha_up: Step increase
            delta_alpha_down: Step decrease
            alpha_min: Minimum authority level
            sigma_alpha: Noise std for small fluctuations
            rng: Random generator for reproducibility
        """
        self.tau_up = tau_up
        self.tau_down = tau_down
        self.d_down = d_down
        self.delta_alpha_up = delta_alpha_up
        self.delta_alpha_down = delta_alpha_down
        self.alpha_min = alpha_min
        self.sigma_alpha = sigma_alpha
        self.rng = rng if rng is not None else np.random.default_rng()

    def update(self, alpha_k: float, ttc: float, distance: float) -> float:
        """
        Classical rule-based update with mild noise.

        alpha_{k+1} = min(1, alpha_k + delta_up)              if TTC < tau_up
                    = max(alpha_min, alpha_k - delta_down)   if TTC > tau_down and d > d_down
                    = alpha_k                                 otherwise
        Then add noise: clip(alpha + N(0, sigma^2), 0, 1)
        """
        if ttc < self.tau_up:
            alpha_next = min(1.0, alpha_k + self.delta_alpha_up)
        elif ttc > self.tau_down and distance > self.d_down:
            alpha_next = max(self.alpha_min, alpha_k - self.delta_alpha_down)
        else:
            alpha_next = alpha_k

        # add small noise
        noise = self.rng.normal(0.0, self.sigma_alpha)
        alpha_next += noise

        return float(np.clip(alpha_next, 0.0, 1.0))


class LLMOnlyAllocator:
    """Baseline 2: LLM-only authority (intent→proposal with inertia, NO monitor)."""

    def __init__(self, eta: float = 0.3):
        """
        Args:
            eta: Proposal inertia [0,1]
        """
        self.eta = eta

    def update(self, alpha_k: float, intent: AuthorityIntent) -> float:
        """
        Intent-to-proposal update with NO projection.

        tilde_alpha_{k+1} = clip(alpha_k + eta*(theta - alpha_k), 0, 1)  INCREASE/DECREASE
                          = alpha_k                                       HOLD/CONFIRM
                          = alpha_k                                       FALLBACK
        """
        if intent.intent_type in ['INCREASE', 'DECREASE']:
            alpha_prop = alpha_k + self.eta * (intent.target_alpha - alpha_k)
        else:  # HOLD, CONFIRM, FALLBACK
            alpha_prop = alpha_k

        return float(np.clip(alpha_prop, 0.0, 1.0))


class NeSyAllocator:
    """NeSy: LLM intent + invariant monitor (rate + emergency monotonicity)."""

    def __init__(self, eta: float, gamma: float, dt: float):
        """
        Args:
            eta: Proposal inertia [0,1]
            gamma: Max authority rate [1/s]
            dt: Sampling period [s]
        """
        self.eta = eta
        self.gamma = gamma
        self.dt = dt

    def update(self, alpha_k: float, intent: AuthorityIntent, emergency: bool) -> float:
        """
        Intent→proposal→projection.

        1. Compute proposal tilde_alpha with inertia
        2. Project to invariants: alpha_{k+1} = Pi_{I_k}(tilde_alpha)
        """
        # intent to proposal
        if intent.intent_type in ['INCREASE', 'DECREASE']:
            alpha_prop = alpha_k + self.eta * (intent.target_alpha - alpha_k)
        else:  # HOLD, CONFIRM, FALLBACK
            alpha_prop = alpha_k

        alpha_prop = np.clip(alpha_prop, 0.0, 1.0)

        # project to invariants
        alpha_next = project_invariants(alpha_prop, alpha_k, emergency, self.dt, self.gamma)

        return alpha_next
