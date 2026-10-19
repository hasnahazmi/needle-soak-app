"""
ai_recipe.py
Lets an AI model propose the soak / alignment RECIPE itself.

Flow
----
1. The app sends the model your raw readings, the curve-fit parameters
   (A, tau, C), the saturation threshold and the stability rule.
   (It does NOT send the curve-fit's own soak time / schedule, so the AI's
   answer is an independent second opinion.)
2. The model must answer with a single JSON object (the recipe).
3. The reply is validated (numbers sane, schedule ascending, ...).
4. The result is cross-checked against the curve-fit calculation and any
   disagreement is flagged for the engineer.

Providers
---------
* "Claude (cloud API)" - needs `pip install anthropic` and an API key.
  Users do NOT need to install anything else (no Ollama).
  NOTE: readings are sent over the internet to the provider.
* "Ollama (local)"     - needs Ollama running locally; data never leaves the PC.
* "Microsoft Copilot (copy/paste)" - no API, no key, no IT setup. The app builds
  the prompt; the user pastes it into Copilot and pastes the reply back.

Nothing in here raises into the GUI: request_ai_recipe() always returns
(recipe, error_message).
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass

import numpy as np

PROVIDER_CLAUDE = "Claude (cloud API)"
PROVIDER_OLLAMA = "Ollama (local)"
PROVIDER_COPILOT = "Microsoft Copilot (copy/paste)"
PROVIDERS = [PROVIDER_COPILOT, PROVIDER_CLAUDE, PROVIDER_OLLAMA]

# Providers that are called automatically by the app. Copilot is manual:
# the user pastes the prompt into Copilot and pastes the reply back.
MANUAL_PROVIDERS = {PROVIDER_COPILOT}

DEFAULT_MODELS = {
    PROVIDER_CLAUDE: "claude-haiku-5-5",   # use "claude-sonnet-5-5" for higher quality
    PROVIDER_OLLAMA: "llama3.2",
    PROVIDER_COPILOT: "",
}

MAX_PROMPT_ROWS = 60       # downsample very long profiles before sending


# ----------------------------------------------------------------------
# Data classes
# ----------------------------------------------------------------------
@dataclass
class AISettings:
    provider: str = PROVIDER_CLAUDE
    model: str = ""
    api_key: str = ""          # never written to disk by the app

    def resolved_model(self) -> str:
        return self.model.strip() or DEFAULT_MODELS.get(self.provider, "")

    def resolved_key(self) -> str:
        return self.api_key.strip() or os.environ.get("ANTHROPIC_API_KEY", "").strip()


@dataclass
class AIRecipe:
    soak_time_min: float
    saturated: bool
    alignment_checks_min: list
    confidence: str
    interpretation: str
    production_recommendation: str
    risk_if_reduced: str
    provider: str = ""
    model: str = ""

    def as_text(self, threshold_um: float, header: list | None = None) -> str:
        bar = "=" * 36
        checks = (", ".join(f"{x:.1f}" for x in self.alignment_checks_min)
                  if self.alignment_checks_min else "(none)")
        lines = [
            bar,
            "AI RECIPE SUGGESTION",
            bar,
            *((list(header) + [""]) if header else []),
            f"Soak time  : {self.soak_time_min:.1f} min",
            f"Saturated  : {'YES' if self.saturated else 'NO'}"
            f"  (threshold {threshold_um:g} um)",
            f"Confidence : {self.confidence or 'n/a'}",
            "",
            "Alignment checks (elapsed min):",
            checks,
            "",
            "Interpretation:",
            self.interpretation or "-",
            "",
            "Production recommendation:",
            self.production_recommendation or "-",
            "",
            "Risk if soak time is reduced:",
            self.risk_if_reduced or "-",
            "",
            f"Source: {self.provider} / {self.model}",
        ]
        return "\n".join(lines)


class AIUnavailable(Exception):
    """Friendly, user-readable reason the AI could not be used."""


# ----------------------------------------------------------------------
# Prompt
# ----------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a senior wafer-probing process engineer who sets needle soak and "
    "alignment recipes for probe cards. You reply with ONE JSON object and "
    "nothing else - no prose before or after it, no markdown fences."
)


def _downsample(t, y, max_rows=MAX_PROMPT_ROWS):
    if len(t) <= max_rows:
        return t, y
    idx = np.unique(np.linspace(0, len(t) - 1, max_rows).round().astype(int))
    return t[idx], y[idx]


def build_prompt(t, y, fit, threshold_um: float, max_stable_slope: float,
                 temperature_c: float | None = None) -> str:
    t, y = _downsample(np.asarray(t, float), np.asarray(y, float))
    rows = "\n".join(f"{ti:.1f}, {yi:.2f}" for ti, yi in zip(t, y))

    if fit is not None and getattr(fit, "success", False):
        fit_txt = (
            "Exponential fit  diff(t) = A*exp(-t/tau) + C  :\n"
            f"  A = {fit.A:.3f} um, tau = {fit.tau:.3f} min, C = {fit.C:.3f} um, "
            f"R2 = {fit.r2:.4f}, RMSE = {fit.rmse:.3f} um"
        )
    else:
        fit_txt = "No reliable curve fit is available (too few / unsuitable points)."

    temp_txt = (f"Soak / chuck temperature: {temperature_c:g} degC.\n"
                if temperature_c is not None else "")

    return f"""Needle height difference vs elapsed soak time (probe card soaking).
{temp_txt}
DATA  (elapsed_min, height_diff_um):
{rows}

{fit_txt}

DEFINITIONS
- Saturation threshold: {threshold_um:g} um.
- The needles count as SATURATED when the height difference is below the
  threshold AND the curve has flattened (|slope| <= {max_stable_slope:g} um/min).
- Soak time = the first elapsed time at which the needles are saturated.
  You may extrapolate with the fit, but never claim saturation if the fit's
  floor C is at or above the threshold.
- Alignment checks = elapsed times (min) at which the operator re-measures
  alignment during the soak. Rule of thumb: spacing 2 min while |slope| > 2
  um/min, 5 min while > 0.5, 10 min while > 0.1, otherwise 15 min. The last
  check should be at the soak time.

Return exactly this JSON object:
{{
  "soak_time_min": <number>,
  "saturated": <true|false>,
  "alignment_checks_min": [<ascending numbers, elapsed minutes>],
  "confidence": "<high|medium|low>",
  "interpretation": "<2-3 sentences on what the data shows>",
  "production_recommendation": "<1-2 sentences>",
  "risk_if_reduced": "<1-2 sentences on the risk of soaking shorter>"
}}
If saturation is NOT reached or predicted, set "saturated": false, give your
best estimate of the extra soak needed in "soak_time_min" (or the last elapsed
time if you cannot estimate), and use an empty alignment_checks_min list."""


# ----------------------------------------------------------------------
# Parsing + validation
# ----------------------------------------------------------------------
def _extract_json(text: str) -> dict:
    text = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S | re.I)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("reply did not contain a JSON object")
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"reply was not valid JSON ({exc.msg})") from exc
    if not isinstance(data, dict):
        raise ValueError("reply JSON was not an object")
    return data


def _to_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in ("true", "yes", "1"):
            return True
        if s in ("false", "no", "0"):
            return False
    raise ValueError(f"'saturated' must be true/false, got {v!r}")


def _to_number(v, name: str) -> float:
    try:
        x = float(v)
    except (TypeError, ValueError):
        raise ValueError(f"'{name}' must be a number, got {v!r}") from None
    if not math.isfinite(x):
        raise ValueError(f"'{name}' must be finite")
    return x


def parse_and_validate(text: str, last_elapsed_min: float) -> AIRecipe:
    """Turn the raw model reply into a checked AIRecipe, or raise ValueError."""
    data = _extract_json(text)

    for key in ("soak_time_min", "saturated"):
        if key not in data:
            raise ValueError(f"missing field '{key}'")

    soak = _to_number(data["soak_time_min"], "soak_time_min")
    limit = max(600.0, 5.0 * last_elapsed_min)
    if not 0.0 <= soak <= limit:
        raise ValueError(f"soak_time_min={soak:g} is outside the plausible range 0-{limit:g}")

    saturated = _to_bool(data["saturated"])

    raw_checks = data.get("alignment_checks_min", []) or []
    if not isinstance(raw_checks, (list, tuple)):
        raise ValueError("'alignment_checks_min' must be a list")
    checks = [_to_number(c, "alignment_checks_min") for c in raw_checks]
    if any(c <= 0 for c in checks):
        raise ValueError("alignment checks must be > 0 min")
    checks = sorted(set(round(c, 1) for c in checks if c <= soak + 0.5))[:60]
    if not saturated:
        checks = []

    return AIRecipe(
        soak_time_min=round(soak, 1),
        saturated=saturated,
        alignment_checks_min=checks,
        confidence=str(data.get("confidence", "")).strip().lower(),
        interpretation=str(data.get("interpretation", "")).strip(),
        production_recommendation=str(data.get("production_recommendation", "")).strip(),
        risk_if_reduced=str(data.get("risk_if_reduced", "")).strip(),
    )


# ----------------------------------------------------------------------
# Cross-check against the curve-fit calculation
# ----------------------------------------------------------------------
def cross_check(ai: AIRecipe, rec, tol_min: float = 5.0, tol_frac: float = 0.2) -> str:
    """One-line verdict comparing the AI recipe with the calculated one."""
    if rec is None:
        return "No curve-fit result to compare with."
    if ai.saturated != rec.saturated:
        return (f"WARNING - AI and curve fit DISAGREE on saturation "
                f"(AI: {'saturated' if ai.saturated else 'not saturated'}, "
                f"curve fit: {'saturated' if rec.saturated else 'not saturated'}). "
                "Verify before using this recipe.")
    if not ai.saturated:
        return "AI and curve fit agree: not saturated yet."
    diff = abs(ai.soak_time_min - rec.soak_time_min)
    limit = max(tol_min, tol_frac * rec.soak_time_min)
    if diff <= limit:
        return (f"OK - AI soak time agrees with the curve-fit calculation "
                f"({rec.soak_time_min:.1f} min, difference {diff:.1f} min).")
    return (f"WARNING - AI soak time {ai.soak_time_min:.1f} min differs from the "
            f"curve-fit calculation ({rec.soak_time_min:.1f} min) by {diff:.1f} min. "
            "Verify before using this recipe.")


# ----------------------------------------------------------------------
# Providers
# ----------------------------------------------------------------------
def _call_provider(settings: AISettings, system: str, user: str) -> str:
    """Return the raw text reply. Raises AIUnavailable with a friendly message."""
    model = settings.resolved_model()

    if settings.provider == PROVIDER_CLAUDE:
        key = settings.resolved_key()
        if not key:
            raise AIUnavailable(
                "No Claude API key. Paste your key into the 'API key' box "
                "(or set the ANTHROPIC_API_KEY environment variable).")
        try:
            import anthropic
        except ImportError:
            raise AIUnavailable(
                "The 'anthropic' package is not installed. Run:  pip install anthropic") from None

        client = anthropic.Anthropic(api_key=key, timeout=60.0, max_retries=1)
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=1200,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except anthropic.AuthenticationError:
            raise AIUnavailable("The Claude API key was rejected. Check the key.") from None
        except anthropic.PermissionDeniedError:
            raise AIUnavailable("This API key is not allowed to use that model.") from None
        except anthropic.NotFoundError:
            raise AIUnavailable(f"Model '{model}' was not found. Check the model name.") from None
        except anthropic.RateLimitError:
            raise AIUnavailable("Claude API rate limit reached. Try again shortly.") from None
        except anthropic.APIConnectionError:
            raise AIUnavailable(
                "Cannot reach the Claude API. Check the internet connection / proxy.") from None
        except anthropic.APIStatusError as exc:
            raise AIUnavailable(f"Claude API error ({exc.status_code}): {exc.message}") from None
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")

    if settings.provider == PROVIDER_OLLAMA:
        try:
            from ollama import chat
        except ImportError:
            raise AIUnavailable(
                "The 'ollama' package is not installed. Run:  pip install ollama") from None
        try:
            resp = chat(
                model=model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                format="json",
                options={"temperature": 0},
            )
        except Exception as exc:  # noqa: BLE001
            raise AIUnavailable(
                f"Ollama request failed: {exc}\nMake sure Ollama is running and the model "
                f"is installed (ollama pull {model}).") from None
        return resp["message"]["content"]

    if settings.provider in MANUAL_PROVIDERS:
        raise AIUnavailable(
            "This provider is copy/paste: use 'Copy prompt' and 'Apply reply'.")

    raise AIUnavailable(f"Unknown AI provider: {settings.provider}")


# ----------------------------------------------------------------------
# Manual (copy/paste) mode, e.g. Microsoft Copilot
# ----------------------------------------------------------------------
def build_manual_prompt(t, y, fit, threshold_um: float, max_stable_slope: float,
                        temperature_c: float | None = None) -> str:
    """One self-contained prompt to paste into a chat window (Copilot etc.)."""
    body = build_prompt(t, y, fit, threshold_um, max_stable_slope, temperature_c)
    return (
        f"{SYSTEM_PROMPT}\n\n{body}\n\n"
        "Put the JSON object in a single code block and write nothing else."
    )


def parse_manual_reply(text: str, last_elapsed_min: float, provider: str = PROVIDER_COPILOT):
    """Validate a reply pasted back from a chat window.
    Returns (AIRecipe | None, error_message)."""
    if not (text or "").strip():
        return None, "Paste Copilot's reply into the box first."
    try:
        recipe = parse_and_validate(text, last_elapsed_min)
    except ValueError as exc:
        return None, (f"Could not use that reply: {exc}.\n"
                      "Ask Copilot: \"Reply again with only the JSON object in a code block.\"")
    recipe.provider = provider
    recipe.model = "pasted reply"
    return recipe, ""


# ----------------------------------------------------------------------
# Public entry point (call from a background thread)
# ----------------------------------------------------------------------
def request_ai_recipe(t, y, fit, threshold_um: float, max_stable_slope: float,
                      settings: AISettings, temperature_c: float | None = None):
    """Ask the AI for a recipe. Returns (AIRecipe | None, error_message)."""
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    if len(t) < 3:
        return None, "Need at least 3 readings for an AI recipe."

    prompt = build_prompt(t, y, fit, threshold_um, max_stable_slope, temperature_c)
    last_t = float(np.max(t))

    last_err = ""
    for attempt in range(2):                      # one automatic retry on a bad reply
        user = prompt if attempt == 0 else (
            f"{prompt}\n\nYour previous reply was rejected: {last_err}. "
            "Reply again with ONLY the JSON object.")
        try:
            raw = _call_provider(settings, SYSTEM_PROMPT, user)
        except AIUnavailable as exc:
            return None, str(exc)
        except Exception as exc:  # noqa: BLE001
            return None, f"Unexpected AI error: {exc}"

        try:
            recipe = parse_and_validate(raw, last_t)
        except ValueError as exc:
            last_err = str(exc)
            continue
        recipe.provider = settings.provider
        recipe.model = settings.resolved_model()
        return recipe, ""

    return None, f"The AI did not return a usable recipe ({last_err})."
