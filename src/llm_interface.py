# src/llm_interface.py

from dataclasses import dataclass
from typing import Optional
import numpy as np
from .vehicle import SafetyMetrics


@dataclass
class AuthorityIntent:
    """LLM-generated authority recommendation."""
    intent_type: str     # {'INCREASE','DECREASE','HOLD','CONFIRM','FALLBACK'}
    target_alpha: float  # desired authority [0,1]
    confidence: float    # [0,1]
    rationale: str       # explanation

    def __post_init__(self):
        assert self.intent_type in ['INCREASE', 'DECREASE', 'HOLD', 'CONFIRM', 'FALLBACK']
        assert 0.0 <= self.target_alpha <= 1.0
        assert 0.0 <= self.confidence <= 1.0


class MockLLM:
    """Rule-based mock LLM with decision table + small noise."""

    def __init__(self, sigma_theta: float = 0.10,
                 rng: Optional[np.random.Generator] = None):
        """
        Args:
            sigma_theta: Target alpha noise std
            rng: Random generator for reproducibility
        """
        self.sigma_theta = sigma_theta
        self.rng = rng if rng is not None else np.random.default_rng()

    def generate_intent(self, metrics: SafetyMetrics, alpha_current: float) -> AuthorityIntent:
        """
        Rule-based intent generation with small perturbation.

        z_k = (INCREASE, 0.90, "Emergency")           if emergency
            = (INCREASE, 0.75, "Tight spacing")       if d < 18.0 and not emergency
            = (HOLD, alpha_k, "Stable")               if 18.0 <= d < 28.0
            = (DECREASE, 0.25, "Safe cruise")         if d >= 28.0

        Add noise: theta <- clip(theta + N(0, sigma^2), 0, 1)
        """
        # decision table
        if metrics.emergency:
            intent_type = 'INCREASE'
            target_alpha = 0.90
            confidence = 0.90
            rationale = "Emergency"
        elif metrics.distance < 18.0:
            intent_type = 'INCREASE'
            target_alpha = 0.75
            confidence = 0.75
            rationale = "Tight spacing"
        elif metrics.distance < 28.0:
            intent_type = 'HOLD'
            target_alpha = alpha_current
            confidence = 0.80
            rationale = "Stable"
        else:
            intent_type = 'DECREASE'
            target_alpha = 0.25
            confidence = 0.70
            rationale = "Safe cruise"

        # add small perturbation to target
        noise = self.rng.normal(0.0, self.sigma_theta)
        target_alpha = float(np.clip(target_alpha + noise, 0.0, 1.0))

        return AuthorityIntent(
            intent_type=intent_type,
            target_alpha=target_alpha,
            confidence=confidence,
            rationale=rationale
        )


def parse_llm_response(response: str, fallback_alpha: float) -> AuthorityIntent:
    """
    Parse LLM JSON response, return FALLBACK intent on failure.

    Expected JSON format:
    {
      "intent_type": "INCREASE|DECREASE|HOLD|CONFIRM|FALLBACK",
      "target_alpha": <float 0..1>,
      "confidence": <float 0..1>,
      "rationale": "<string>"
    }

    On parse failure or invalid values, returns FALLBACK with safe HOLD.
    """
    import json

    try:
        data = json.loads(response)
        # clip values to valid range before validation
        target_alpha = float(np.clip(data['target_alpha'], 0.0, 1.0))
        confidence = float(np.clip(data['confidence'], 0.0, 1.0))

        intent = AuthorityIntent(
            intent_type=data['intent_type'],
            target_alpha=target_alpha,
            confidence=confidence,
            rationale=data['rationale']
        )
        return intent
    except (json.JSONDecodeError, KeyError, ValueError, AssertionError):
        # fallback on any parsing or validation error
        return AuthorityIntent(
            intent_type='FALLBACK',
            target_alpha=fallback_alpha,
            confidence=0.0,
            rationale='Parse failure - safe HOLD'
        )
