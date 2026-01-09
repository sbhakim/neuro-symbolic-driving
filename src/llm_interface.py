# src/llm_interface.py

from dataclasses import dataclass
from typing import Optional, Dict
import os
import hashlib
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


class RealLLM:
    """
    Thin OpenAI-backed LLM wrapper (drop-in replacement for MockLLM).

    - Reads API key from OPENAI_API_KEY (do not hardcode keys).
    - Uses deterministic decoding (temperature=0) for reproducibility.
    - Includes tiny in-memory caching to stabilize repeated runs.
    - Returns FALLBACK intent if the API is unavailable or output is invalid.
    """

    def __init__(self,
                 model: str = "gpt-4o-mini",
                 temperature: float = 0.0,
                 max_output_tokens: int = 120,
                 min_confidence: float = 0.50,
                 enable_cache: bool = True):
        self.model = model
        self.temperature = float(temperature)
        self.max_output_tokens = int(max_output_tokens)
        self.min_confidence = float(min_confidence)
        self.enable_cache = bool(enable_cache)
        self._cache: Dict[str, str] = {}

        # Lazy client init so importing this module doesn't require openai installed.
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

        # If key is missing, we will fail closed (fallback).
        if not os.getenv("OPENAI_API_KEY"):
            return None

        try:
            from openai import OpenAI  # official SDK
        except Exception:
            return None

        try:
            self._client = OpenAI()
            return self._client
        except Exception:
            return None

    @staticmethod
    def _prompt(metrics: SafetyMetrics, alpha_current: float) -> str:
        # Keep prompt short and strictly structured to reduce hallucinated fields.
        return (
            "You are an authority allocation assistant for shared-control driving.\n"
            "Return ONLY a single-line JSON object with keys:\n"
            "intent_type (INCREASE|DECREASE|HOLD|CONFIRM|FALLBACK), target_alpha (0..1), "
            "confidence (0..1), rationale (string).\n\n"
            f"Signals:\n"
            f"- distance_m: {metrics.distance:.3f}\n"
            f"- ttc_s: {metrics.ttc:.3f}\n"
            f"- emergency: {bool(metrics.emergency)}\n"
            f"- alpha_current: {float(alpha_current):.3f}\n\n"
            "Guidance:\n"
            "- If emergency is true OR ttc_s is very low, prefer INCREASE with higher target_alpha.\n"
            "- If distance is large and ttc_s is high, prefer DECREASE.\n"
            "- Otherwise HOLD.\n"
        )

    def generate_intent(self, metrics: SafetyMetrics, alpha_current: float) -> AuthorityIntent:
        prompt = self._prompt(metrics, alpha_current)
        cache_key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

        if self.enable_cache and cache_key in self._cache:
            return parse_llm_response(self._cache[cache_key], fallback_alpha=alpha_current)

        client = self._get_client()
        if client is None:
            return AuthorityIntent(
                intent_type='FALLBACK',
                target_alpha=float(np.clip(alpha_current, 0.0, 1.0)),
                confidence=0.0,
                rationale='OpenAI client unavailable - safe HOLD'
            )

        try:
            resp = client.responses.create(
                model=self.model,
                input=prompt,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            )
            text = getattr(resp, "output_text", None)
            if text is None:
                # Defensive: some SDK versions may expose output differently
                text = str(resp)

            if self.enable_cache:
                self._cache[cache_key] = text

            intent = parse_llm_response(text, fallback_alpha=alpha_current)

            # Minimal extra guardrail: low-confidence => safe HOLD (keeps paper defensible)
            if intent.intent_type not in ['FALLBACK'] and intent.confidence < self.min_confidence:
                return AuthorityIntent(
                    intent_type='HOLD',
                    target_alpha=float(np.clip(alpha_current, 0.0, 1.0)),
                    confidence=float(np.clip(intent.confidence, 0.0, 1.0)),
                    rationale='Low confidence - safe HOLD'
                )

            return intent

        except Exception:
            return AuthorityIntent(
                intent_type='FALLBACK',
                target_alpha=float(np.clip(alpha_current, 0.0, 1.0)),
                confidence=0.0,
                rationale='OpenAI call failed - safe HOLD'
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
