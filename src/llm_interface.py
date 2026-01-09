# src/llm_interface.py

from dataclasses import dataclass
from typing import Optional, Dict, Any
import os
import hashlib
import time
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
    - Uses deterministic decoding (temperature=0) for reproducibility (when supported by the model).
    - Includes tiny in-memory caching to stabilize repeated runs.
    - Returns FALLBACK intent if the API is unavailable or output is invalid.
    - Optional verbose logging to confirm OpenAI usage during simulation.
    """

    def __init__(self,
                 model: str = "gpt-5-nano",
                 temperature: float = 0.0,
                 max_output_tokens: int = 120,
                 min_confidence: float = 0.50,
                 enable_cache: bool = True,
                 verbose: bool = False):
        self.model = model
        self.temperature = float(temperature)
        self.max_output_tokens = int(max_output_tokens)
        self.min_confidence = float(min_confidence)
        self.enable_cache = bool(enable_cache)
        self.verbose = bool(verbose)
        self._cache: Dict[str, str] = {}
        self._client = None

    def _get_client(self):
        if self._client is not None:
            return self._client

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
        return (
            "You are an authority allocation assistant for shared-control driving.\n"
            "Return ONLY a single-line JSON object. No markdown, no extra text.\n"
            "Keys:\n"
            "- intent_type: INCREASE|DECREASE|HOLD|CONFIRM|FALLBACK\n"
            "- target_alpha: number in [0,1]\n"
            "- confidence: number in [0,1]\n"
            "- rationale: short string (<= 8 words)\n\n"
            f"Signals:\n"
            f"distance_m={metrics.distance:.3f}\n"
            f"ttc_s={metrics.ttc:.3f}\n"
            f"emergency={bool(metrics.emergency)}\n"
            f"alpha_current={float(alpha_current):.3f}\n\n"
            "Guidance:\n"
            "If emergency=true OR ttc_s is very low => INCREASE with higher target_alpha.\n"
            "If distance_m is large and ttc_s is high => DECREASE.\n"
            "Otherwise => HOLD.\n"
        )

    @staticmethod
    def _extract_text(resp) -> str:
        """
        Robustly extract the model's textual output from OpenAI Responses API objects.

        Different SDK versions expose either:
        - resp.output_text (string), or
        - resp.output[*].content[*].text (structured), or
        - dict-like equivalents.
        """
        text = getattr(resp, "output_text", None)
        if isinstance(text, str) and text.strip():
            return text

        try:
            chunks = []
            output = getattr(resp, "output", None)
            if output is None and isinstance(resp, dict):
                output = resp.get("output", None)

            for item in output or []:
                content = getattr(item, "content", None)
                if content is None and isinstance(item, dict):
                    content = item.get("content", None)

                for c in content or []:
                    t = getattr(c, "text", None)
                    if t is None and isinstance(c, dict):
                        t = c.get("text", None)
                    if isinstance(t, str) and t.strip():
                        chunks.append(t)

            if chunks:
                return "\n".join(chunks)
        except Exception:
            pass

        return str(resp)

    @staticmethod
    def _incomplete_max_tokens(resp) -> bool:
        """
        Detect the 'incomplete because max_output_tokens' condition.
        """
        status = getattr(resp, "status", None)
        if status is None and isinstance(resp, dict):
            status = resp.get("status", None)

        inc = getattr(resp, "incomplete_details", None)
        if inc is None and isinstance(resp, dict):
            inc = resp.get("incomplete_details", None)

        reason = getattr(inc, "reason", None) if inc is not None else None
        if reason is None and isinstance(inc, dict):
            reason = inc.get("reason", None)

        return (status == "incomplete") and (reason == "max_output_tokens")

    def generate_intent(self, metrics: SafetyMetrics, alpha_current: float) -> AuthorityIntent:
        prompt = self._prompt(metrics, alpha_current)
        cache_key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

        if self.enable_cache and cache_key in self._cache:
            if self.verbose:
                print(f"[RealLLM] cache hit (model={self.model})")
            return parse_llm_response(self._cache[cache_key], fallback_alpha=alpha_current)

        client = self._get_client()
        if client is None:
            if self.verbose:
                print("[RealLLM] FALLBACK: OpenAI client unavailable (missing key/sdk/init)")
            return AuthorityIntent(
                intent_type='FALLBACK',
                target_alpha=float(np.clip(alpha_current, 0.0, 1.0)),
                confidence=0.0,
                rationale='OpenAI client unavailable - safe HOLD'
            )

        t0 = time.time()
        if self.verbose:
            print(f"[RealLLM] calling OpenAI (model={self.model}, emergency={bool(metrics.emergency)})")

        def _call(max_out: int):
            # NOTE: Some OpenAI model families (e.g., gpt-5*) reject sampling params like temperature.
            req: Dict[str, Any] = {
                "model": self.model,
                "input": prompt,
                "max_output_tokens": int(max_out),
                # Force JSON output to stabilize parsing (no markdown, no extra text).
                "text": {"format": {"type": "json_object"}},
            }

            # GPT-5 family: set minimal reasoning effort (gpt-5-nano does NOT accept 'none').
            if str(self.model).startswith("gpt-5"):
                req["reasoning"] = {"effort": "minimal"}
            else:
                req["temperature"] = self.temperature

            return client.responses.create(**req)

        try:
            resp = _call(self.max_output_tokens)

            # If the response was cut off due to token limit, retry once with a larger budget.
            if self._incomplete_max_tokens(resp):
                if self.verbose:
                    print("[RealLLM] retry: response incomplete due to max_output_tokens")
                bumped = max(int(self.max_output_tokens) * 4, 512)
                bumped = min(bumped, 2048)
                resp = _call(bumped)

            text = self._extract_text(resp)

            if self.enable_cache:
                self._cache[cache_key] = text

            intent = parse_llm_response(text, fallback_alpha=alpha_current)

            if self.verbose and intent.intent_type == "FALLBACK":
                preview = str(text).replace("\n", " ")[:240]
                print(f"[RealLLM] parsed FALLBACK; raw preview: {preview}")

            if intent.intent_type not in ['FALLBACK'] and intent.confidence < self.min_confidence:
                if self.verbose:
                    dt = time.time() - t0
                    print(f"[RealLLM] low confidence ({intent.confidence:.2f}) -> HOLD (dt={dt:.3f}s)")
                return AuthorityIntent(
                    intent_type='HOLD',
                    target_alpha=float(np.clip(alpha_current, 0.0, 1.0)),
                    confidence=float(np.clip(intent.confidence, 0.0, 1.0)),
                    rationale='Low confidence - safe HOLD'
                )

            if self.verbose:
                dt = time.time() - t0
                print(f"[RealLLM] ok: {intent.intent_type} α={intent.target_alpha:.2f} c={intent.confidence:.2f} (dt={dt:.3f}s)")
            return intent

        except Exception as e:
            if self.verbose:
                dt = time.time() - t0
                print(f"[RealLLM] FALLBACK: OpenAI call failed (dt={dt:.3f}s): {type(e).__name__}: {e}")
            return AuthorityIntent(
                intent_type='FALLBACK',
                target_alpha=float(np.clip(alpha_current, 0.0, 1.0)),
                confidence=0.0,
                rationale='OpenAI call failed - safe HOLD'
            )


class OfflineHFLLM:
    """
    Offline Hugging Face LLM wrapper (drop-in replacement for MockLLM).

    - Loads model from local HF cache (local_files_only=True).
    - Lazy loads once per process.
    - Deterministic generation (do_sample=False).
    - Uses tiny in-memory caching.
    - Fails closed to FALLBACK on any load/generation/parse error.
    """

    def __init__(self,
                 model_id: str,
                 max_new_tokens: int = 120,
                 enable_cache: bool = True,
                 verbose: bool = False,
                 device_map: str = "auto",
                 use_4bit_if_available: bool = True):
        self.model_id = model_id
        self.max_new_tokens = int(max_new_tokens)
        self.enable_cache = bool(enable_cache)
        self.verbose = bool(verbose)
        self.device_map = device_map
        self.use_4bit_if_available = bool(use_4bit_if_available)

        self._cache: Dict[str, str] = {}
        self._tok = None
        self._model = None

    @staticmethod
    def _prompt(metrics: SafetyMetrics, alpha_current: float) -> str:
        # Keep consistent with RealLLM to keep results comparable.
        return (
            "Return ONLY a single-line JSON object. No markdown. No extra text.\n"
            "Keys: intent_type, target_alpha, confidence, rationale.\n"
            "intent_type must be one of INCREASE|DECREASE|HOLD|CONFIRM|FALLBACK.\n"
            "target_alpha and confidence must be numbers in [0,1].\n\n"
            f"distance_m={metrics.distance:.3f}\n"
            f"ttc_s={metrics.ttc:.3f}\n"
            f"emergency={bool(metrics.emergency)}\n"
            f"alpha_current={float(alpha_current):.3f}\n"
        )

    def _get_model(self):
        if self._model is not None and self._tok is not None:
            return self._tok, self._model

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
        except Exception:
            return None, None

        t0 = time.time()
        if self.verbose:
            print(f"[OfflineHFLLM] loading {self.model_id} (offline)")

        # Try 4-bit if available (keeps it feasible on smaller GPUs). Fall back gracefully.
        quant_cfg = None
        if self.use_4bit_if_available:
            try:
                import torch
                from transformers import BitsAndBytesConfig
                quant_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
            except Exception:
                quant_cfg = None

        try:
            self._tok = AutoTokenizer.from_pretrained(self.model_id, local_files_only=True)
            kwargs: Dict[str, Any] = {
                "local_files_only": True,
                "device_map": self.device_map,
            }
            if quant_cfg is not None:
                kwargs["quantization_config"] = quant_cfg

            self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)

            if self.verbose:
                dt = time.time() - t0
                print(f"[OfflineHFLLM] loaded {self.model_id} in {dt:.3f}s")
            return self._tok, self._model

        except Exception:
            self._tok = None
            self._model = None
            return None, None

    def generate_intent(self, metrics: SafetyMetrics, alpha_current: float) -> AuthorityIntent:
        prompt = self._prompt(metrics, alpha_current)
        cache_key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()

        if self.enable_cache and cache_key in self._cache:
            if self.verbose:
                print(f"[OfflineHFLLM] cache hit ({self.model_id})")
            return parse_llm_response(self._cache[cache_key], fallback_alpha=alpha_current)

        tok, model = self._get_model()
        if tok is None or model is None:
            if self.verbose:
                print("[OfflineHFLLM] FALLBACK: model/tokenizer unavailable")
            return AuthorityIntent(
                intent_type="FALLBACK",
                target_alpha=float(np.clip(alpha_current, 0.0, 1.0)),
                confidence=0.0,
                rationale="Offline model unavailable - safe HOLD",
            )

        try:
            t0 = time.time()
            if self.verbose:
                print(f"[OfflineHFLLM] generate ({self.model_id}, emergency={bool(metrics.emergency)})")

            inputs = tok(prompt, return_tensors="pt")
            # Move to model device if possible
            try:
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
            except Exception:
                pass

            gen_kwargs = dict(
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
            if tok.eos_token_id is not None:
                gen_kwargs["eos_token_id"] = tok.eos_token_id
            if tok.pad_token_id is not None:
                gen_kwargs["pad_token_id"] = tok.pad_token_id

            out = model.generate(**inputs, **gen_kwargs)
            text = tok.decode(out[0], skip_special_tokens=True)

            if self.enable_cache:
                self._cache[cache_key] = text

            intent = parse_llm_response(text, fallback_alpha=alpha_current)

            if self.verbose:
                dt = time.time() - t0
                print(f"[OfflineHFLLM] ok: {intent.intent_type} α={intent.target_alpha:.2f} c={intent.confidence:.2f} (dt={dt:.3f}s)")
            return intent

        except Exception as e:
            if self.verbose:
                print(f"[OfflineHFLLM] FALLBACK: generation failed: {type(e).__name__}")
            return AuthorityIntent(
                intent_type="FALLBACK",
                target_alpha=float(np.clip(alpha_current, 0.0, 1.0)),
                confidence=0.0,
                rationale="Offline generation failed - safe HOLD",
            )


def build_llm(mode: str,
              model: str = "",
              rng: Optional[np.random.Generator] = None,
              verbose: bool = False,
              **kwargs) -> object:
    """
    Factory to build an LLM backend.

    Args:
        mode: 'mock' | 'openai' | 'hf'
        model: model name/id (OpenAI model for 'openai', HF repo id for 'hf')
        rng: RNG for MockLLM
        verbose: verbose logging (RealLLM/OfflineHFLLM)
        kwargs: forwarded to backend constructors

    Returns:
        An object with generate_intent(metrics, alpha_current) -> AuthorityIntent
    """
    mode = (mode or "mock").lower().strip()

    if mode == "mock":
        return MockLLM(rng=rng, **{k: v for k, v in kwargs.items() if k in ["sigma_theta"]})

    if mode == "openai":
        if not model:
            model = "gpt-5-nano"
        return RealLLM(model=model, verbose=verbose, **kwargs)

    if mode == "hf":
        if not model:
            raise ValueError("HF mode requires a Hugging Face model_id (e.g., meta-llama/Llama-3.1-8B)")
        return OfflineHFLLM(model_id=model, verbose=verbose, **kwargs)

    raise ValueError(f"Unknown LLM mode: {mode} (expected: mock|openai|hf)")


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

    def _try_parse(s: str) -> AuthorityIntent:
        data = json.loads(s)
        target_alpha = float(np.clip(data['target_alpha'], 0.0, 1.0))
        confidence = float(np.clip(data['confidence'], 0.0, 1.0))
        return AuthorityIntent(
            intent_type=data['intent_type'],
            target_alpha=target_alpha,
            confidence=confidence,
            rationale=data['rationale']
        )

    try:
        return _try_parse(response)
    except (json.JSONDecodeError, KeyError, ValueError, AssertionError, TypeError):
        # Try to recover if the model wrapped JSON with extra text.
        try:
            s = str(response)
            i = s.find("{")
            j = s.rfind("}")
            if i != -1 and j != -1 and j > i:
                return _try_parse(s[i:j + 1])
        except Exception:
            pass

        return AuthorityIntent(
            intent_type='FALLBACK',
            target_alpha=float(np.clip(fallback_alpha, 0.0, 1.0)),
            confidence=0.0,
            rationale='Parse failure - safe HOLD'
        )
