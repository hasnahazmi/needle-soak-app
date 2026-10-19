"""
analysis.py
Core analysis engine for probe needle soaking / alignment recipe generation.

Model: needle height diff (um) vs elapsed soak time (min) typically follows
an exponential saturation curve:

    diff(t) = A * exp(-t / tau) + C

Where:
    A   = initial amplitude of movement
    tau = time constant (how fast it settles)
    C   = residual / floor value the diff settles at

From the fitted curve we derive:
    - recommended soak time: first t where diff(t) < saturation_threshold
      (and the curve is flattening out)
    - recommended alignment check schedule: tighter while the slope is steep,
      wider as it flattens

The optional AI commentary (local Ollama model) lives in ask_ollama() and is
kept separate from the maths so the app never freezes or fails without it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np
from scipy.optimize import curve_fit


def _exp_decay(t, A, tau, C):
    return A * np.exp(-t / tau) + C


# ============================================================
# Data classes
# ============================================================
@dataclass
class Reading:
    elapsed_min: float
    height_diff_um: float


@dataclass
class FitResult:
    success: bool
    A: float = 0.0
    tau: float = 0.0
    C: float = 0.0
    r2: float = 0.0
    rmse: float = 0.0
    message: str = ""

    def predict(self, t):
        return _exp_decay(np.asarray(t, dtype=float), self.A, self.tau, self.C)


@dataclass
class Recommendation:
    soak_time_min: float
    saturated: bool
    alignment_intervals_min: list
    notes: str
    fit: FitResult | None = None


# ============================================================
# Recipe report (text shown in the GUI / exported)
# ============================================================
@dataclass
class RecipeReport:
    probe_card: str
    wafer_id: str
    generated_time: str

    soak_time_min: float
    saturated: bool
    alignment_checks: list

    A: float
    tau: float
    C: float
    threshold_um: float
    temperature_c: float | None = None

    def as_text(self) -> str:
        bar = "=" * 36
        sub = "-" * 36
        checks = (", ".join(f"{x:.1f}" for x in self.alignment_checks)
                  if self.alignment_checks else "(none)")

        lines = [
            bar,
            "Needle Soak & Alignment Recipe",
            bar,
            f"Probe Card : {self.probe_card}",
            f"Lot / Wafer: {self.wafer_id}",
            f"Temperature: {self.temperature_c:g} °C" if self.temperature_c is not None
            else "Temperature: (not set)",
            f"Generated  : {self.generated_time}",
            "",
            sub,
            "RECOMMENDED RECIPE",
            sub,
            f"Soak Time : {self.soak_time_min:.1f} min",
            f"Saturated : {'YES' if self.saturated else 'NO'}",
            "",
            "Alignment checks (elapsed min):",
            checks,
            "",
            sub,
            "FIT PARAMETERS",
            sub,
            f"A   = {self.A:.1f} µm",
            f"Tau = {self.tau:.1f} min",
            f"C   = {self.C:.1f} µm",
            f"Threshold = {self.threshold_um:.1f} µm",
            "",
            sub,
            "INTERPRETATION",
            sub,
        ]

        if self.saturated:
            lines += [
                f"Height difference drops below {self.threshold_um:.1f} µm",
                f"at approximately {self.soak_time_min:.1f} min.",
                "Needle movement has stabilized.",
                "",
                "Recommended:",
                f"  ✓ Soak for {self.soak_time_min:.1f} min",
                "  ✓ Start production",
                "  ✓ Follow alignment schedule",
            ]
        else:
            lines += [
                f"Height difference has NOT reached {self.threshold_um:.1f} µm.",
                "Needle movement is not yet stable.",
                "",
                "Recommended:",
                "  ✗ Do not start production yet",
                "  ✓ Continue soaking / collect more readings",
            ]
        lines.append(bar)
        return "\n".join(lines)


# ============================================================
# Analysis engine
# ============================================================
class SoakAnalysisEngine:
    """
    Accumulates (time, height_diff) readings and produces a live curve fit
    plus a soak-time / alignment-interval recommendation.
    """

    def __init__(self, saturation_threshold_um: float = 10.0,
                 min_points_for_fit: int = 4,
                 max_stable_slope_um_per_min: float = 1.0):
        self.readings: list[Reading] = []
        self.saturation_threshold_um = saturation_threshold_um
        self.min_points_for_fit = min_points_for_fit
        # curve must also be flatter than this to count as "stable"
        self.max_stable_slope_um_per_min = max_stable_slope_um_per_min

    # ---------------------------------------------------------- data
    def add_reading(self, elapsed_min: float, height_diff_um: float):
        self.readings.append(Reading(float(elapsed_min), float(height_diff_um)))

    def clear(self):
        self.readings.clear()

    def as_arrays(self):
        """Readings as numpy arrays, sorted by time."""
        t = np.array([r.elapsed_min for r in self.readings], dtype=float)
        y = np.array([r.height_diff_um for r in self.readings], dtype=float)
        order = np.argsort(t, kind="stable")
        return t[order], y[order]

    # ----------------------------------------------------------- fit
    def fit(self) -> FitResult:
        t, y = self.as_arrays()

        if len(t) < self.min_points_for_fit:
            return FitResult(success=False, message="Not enough points")

        A0 = max(y[0] - y[-1], 1.0)
        tau0 = max(t[-1] / 3.0, 1.0)
        C0 = max(y[-1], 0.0)

        try:
            popt, _ = curve_fit(
                _exp_decay, t, y,
                p0=[A0, tau0, C0],
                bounds=([0, 0.01, 0], [np.inf, np.inf, np.inf]),
                maxfev=5000,
            )
            A, tau, C = (float(v) for v in popt)

            y_pred = _exp_decay(t, A, tau, C)
            ss_res = float(np.sum((y - y_pred) ** 2))
            ss_tot = float(np.sum((y - np.mean(y)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            rmse = float(np.sqrt(np.mean((y - y_pred) ** 2)))

            return FitResult(success=True, A=A, tau=tau, C=C, r2=r2, rmse=rmse)

        except Exception as exc:  # noqa: BLE001
            return FitResult(success=False, message=f"Fit failed: {exc}")

    # ------------------------------------------------- recommendation
    def recommend(self, horizon_min: float = 120.0,
                  resolution_min: float = 0.5) -> Recommendation:
        fit_result = self.fit()
        t, y = self.as_arrays()
        thr = self.saturation_threshold_um

        # ---- Fallback: no usable fit -> use raw data only ----
        if not fit_result.success:
            below = np.where(y < thr)[0] if len(y) else []
            if len(below) == 0:
                return Recommendation(
                    soak_time_min=float(t[-1]) if len(t) else 0.0,
                    saturated=False,
                    alignment_intervals_min=[],
                    notes="Curve fit unavailable — using raw last reading. "
                          "Collect more points for a reliable recommendation.",
                    fit=None,
                )
            soak_t = float(t[below[0]])
            return Recommendation(
                soak_time_min=round(soak_t, 1),
                saturated=True,
                alignment_intervals_min=self._intervals_from_raw(t, soak_t),
                notes="Curve fit unavailable — recommendation based on raw data only.",
                fit=None,
            )

        # ---- Use fitted curve: first time it is below threshold AND flat enough ----
        horizon = max(horizon_min, 3.0 * float(t[-1]))
        t_grid = np.arange(0.0, horizon + resolution_min, resolution_min)
        y_grid = fit_result.predict(t_grid)
        slope_grid = np.abs(np.gradient(y_grid, t_grid))

        stable_idx = np.where(
            (y_grid < thr) & (slope_grid <= self.max_stable_slope_um_per_min)
        )[0]

        quality = (f"Fit quality: R²={fit_result.r2:.3f}, "
                   f"RMSE={fit_result.rmse:.2f} µm.")
        if fit_result.r2 < 0.9:
            quality += " Fit is weak — treat the result with caution."

        if len(stable_idx) == 0:
            return Recommendation(
                soak_time_min=float(t[-1]),
                saturated=False,
                alignment_intervals_min=[],
                notes=f"Curve has not reached the {thr:g} µm threshold within "
                      f"{horizon:.0f} min (floor C={fit_result.C:.1f} µm, "
                      f"tau={fit_result.tau:.1f} min). Consider extending the "
                      f"soak. {quality}",
                fit=fit_result,
            )

        soak_time = float(t_grid[stable_idx[0]])
        intervals = self._intervals_from_fit(fit_result, soak_time)

        return Recommendation(
            soak_time_min=round(soak_time, 1),
            saturated=True,
            alignment_intervals_min=intervals,
            notes=f"Predicted to fall below {thr:g} µm at {soak_time:.1f} min. {quality}",
            fit=fit_result,
        )

    # ----------------------------------------------------- schedules
    def _intervals_from_fit(self, fit_result: FitResult, soak_time: float) -> list:
        """
        Adaptive alignment schedule: tight while |slope| is steep, widening
        as the curve flattens, stopping at soak_time (saturation).
        """
        schedule = []
        t = 0.0
        while t < soak_time:
            slope = self._local_slope(fit_result, t)
            interval = self._slope_to_interval(slope)
            next_t = min(t + interval, soak_time)
            schedule.append(round(next_t, 1))
            t = next_t
        return schedule

    @staticmethod
    def _intervals_from_raw(t, soak_time) -> list:
        # simple fallback: reuse the measured times up to saturation
        return [round(float(ti), 1) for ti in t if 0 < ti <= soak_time]

    @staticmethod
    def _local_slope(fit_result: FitResult, t: float, dt: float = 0.5) -> float:
        y1 = fit_result.predict([t])[0]
        y2 = fit_result.predict([t + dt])[0]
        return abs((y2 - y1) / dt)

    @staticmethod
    def _slope_to_interval(slope_um_per_min: float) -> float:
        """
        Map local rate-of-change to a suggested alignment check interval.
        Tune these breakpoints against your process data.
        """
        if slope_um_per_min > 2.0:
            return 2.0      # moving fast -> check every 2 min
        elif slope_um_per_min > 0.5:
            return 5.0      # moderate    -> every 5 min
        elif slope_um_per_min > 0.1:
            return 10.0     # slowing     -> every 10 min
        else:
            return 15.0     # nearly flat -> every 15 min


# ============================================================
# Optional AI commentary (local Ollama model)
# ============================================================
def ask_ollama(fit: FitResult, soak_time: float, intervals: list,
               threshold_um: float, model: str = "llama3.2") -> str:
    """
    Ask a local Ollama model to comment on the result. Never raises:
    returns a readable message if Ollama is missing / not running.
    Call this from a background thread (it can take several seconds).
    """
    prompt = f"""You are a wafer probing process engineer.

Curve fit results (needle height diff = A*exp(-t/tau) + C):
  A   = {fit.A:.2f} um
  Tau = {fit.tau:.2f} min
  C   = {fit.C:.2f} um
  Saturation threshold = {threshold_um:.1f} um

Recommended soak time: {soak_time:.1f} min
Alignment check schedule (elapsed min): {intervals}

Provide:
1. Process interpretation
2. Stability assessment
3. Production recommendation
4. Risk if soak time is reduced

Keep the response concise."""

    try:
        from ollama import chat
    except ImportError:
        return ("AI recommendation unavailable: the 'ollama' Python package "
                "is not installed.\nRun:  pip install ollama")

    try:
        response = chat(model=model, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"].strip()
    except Exception as exc:  # noqa: BLE001
        return (f"AI recommendation unavailable: {exc}\n\n"
                f"Make sure Ollama is installed and running, and the model "
                f"is downloaded:\n  ollama pull {model}")


def timestamp_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")
