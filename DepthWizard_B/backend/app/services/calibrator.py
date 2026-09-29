"""
backend/app/services/calibrator.py
────────────────────────────────────
Robust dual-mode calibration engine for DepthWizard.

Pipeline (PRD Section 9.2)
──────────────────────────
1. Sign check  — Pearson r(d̂, GCP_z) over train-split anchors.
                 If r < 0 → flip depth (d̂ ← 1.0 − d̂).
2. Affine fit  — RANSAC: Z = s·d̂ + t, constrained to s > 0.
                 Falls back to OLS if RANSAC fails (e.g. too few points / flat relief).
3. Validation  — r, MAE, RMSE computed *exclusively* against holdout-split anchors
                 (never the points used for fitting → honest HUD numbers).

Public API
──────────
    result: CalibrationResult = calibrate(
        depth_map   : np.ndarray,          # HxW float32 [0,1]
        train_gcps  : list[GCPAnchor],
        holdout_gcps: list[GCPAnchor],
    )
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import stats  # type: ignore

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GCPAnchor:
    """A single ground-control point anchor."""
    pixel_x: int
    pixel_y: int
    elevation_m: float
    split: str      # 'train' | 'holdout'


@dataclass
class CalibrationResult:
    """Returned by :func:`calibrate`."""
    # Applied depth (potentially inverted)
    depth_map: np.ndarray               # HxW float32 [0,1] — post sign-correction

    # Affine parameters
    affine_s: float                     # scale  (always > 0 post-correction)
    affine_t: float                     # offset

    # Metadata
    depth_inverted: bool                # True if raw depth was sign-flipped
    fit_method: str                     # 'ransac' | 'ols'

    # Validation metrics (holdout only) — None if insufficient holdout points
    metric_r: Optional[float] = None
    metric_mae: Optional[float] = None
    metric_rmse: Optional[float] = None

    # Predicted height matrix (metric AMSL)
    height_matrix: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_MIN_RANSAC_POINTS = 3       # sklearn RANSAC needs at least this many samples
_MIN_VALIDATION_POINTS = 2   # need ≥2 for Pearson r to be meaningful


def _sample_depth(depth_map: np.ndarray, gcps: list[GCPAnchor]) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract depth values at GCP pixel locations.
    Clips coordinates to valid array bounds to tolerate minor off-by-one errors.

    Returns:
        d_vals : float32 array of sampled depth values
        z_vals : float32 array of ground-truth elevations
    """
    h, w = depth_map.shape[:2]
    d_vals, z_vals = [], []
    for gcp in gcps:
        row = int(np.clip(gcp.pixel_y, 0, h - 1))
        col = int(np.clip(gcp.pixel_x, 0, w - 1))
        d_vals.append(float(depth_map[row, col]))
        z_vals.append(float(gcp.elevation_m))
    return np.array(d_vals, dtype=np.float32), np.array(z_vals, dtype=np.float32)


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Return Pearson r; 0.0 on degenerate inputs."""
    if len(x) < 2:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r, _ = stats.pearsonr(x.astype(float), y.astype(float))
    return float(r) if np.isfinite(r) else 0.0


def _ransac_affine(
    d_vals: np.ndarray,
    z_vals: np.ndarray,
) -> tuple[float, float, str]:
    """
    Fit  Z = s·d + t  using RANSAC with s > 0 constraint.

    Strategy for s > 0 enforcement:
      We try RANSAC normally.  If the fitted s ≤ 0 (pathological case on very
      flat terrain where RANSAC picks a near-constant inlier set), we fall back
      to OLS.  In both cases we force s = max(s, 1e-6) so the downstream
      pipeline never divides by zero or produces NaN.

    Returns:
        (s, t, fit_method_label)
    """
    try:
        from sklearn.linear_model import RANSACRegressor  # type: ignore
        from sklearn.linear_model import LinearRegression  # type: ignore

        X = d_vals.reshape(-1, 1).astype(float)
        y = z_vals.astype(float)

        ransac = RANSACRegressor(
            estimator=LinearRegression(fit_intercept=True),
            min_samples=max(_MIN_RANSAC_POINTS, int(0.5 * len(d_vals))),
            residual_threshold=float(np.std(y) * 0.5) if np.std(y) > 1.0 else 10.0,
            max_trials=500,
            random_state=42,
        )
        ransac.fit(X, y)

        s = float(ransac.estimator_.coef_[0])
        t = float(ransac.estimator_.intercept_)

        if s <= 0:
            log.warning(
                "RANSAC returned s=%.4f ≤ 0 — depth sign-check should have caught this. "
                "Falling back to OLS.",
                s,
            )
            s, t = _ols_affine(d_vals, z_vals)
            return max(s, 1e-6), t, "ols_fallback"

        log.debug("RANSAC fit: s=%.4f, t=%.4f", s, t)
        return s, t, "ransac"

    except Exception as exc:
        log.warning("RANSAC failed (%s) — falling back to OLS.", exc)
        s, t = _ols_affine(d_vals, z_vals)
        return max(s, 1e-6), t, "ols"


def _ols_affine(
    d_vals: np.ndarray,
    z_vals: np.ndarray,
) -> tuple[float, float]:
    """Ordinary Least Squares affine fit: Z = s·d + t."""
    if len(d_vals) < 2:
        # Degenerate: single point — can only fix offset, set s=1
        s = 1.0
        t = float(z_vals[0]) - float(d_vals[0])
        return s, t

    # Manual OLS via numpy for zero-dependency fallback
    d_mean = float(np.mean(d_vals))
    z_mean = float(np.mean(z_vals))
    ss_dd = float(np.sum((d_vals - d_mean) ** 2))

    if ss_dd < 1e-12:
        # Flat depth map — only offset is estimable
        return 1.0, z_mean - d_mean

    s = float(np.sum((d_vals - d_mean) * (z_vals - z_mean)) / ss_dd)
    t = z_mean - s * d_mean
    return max(s, 1e-6), t


def _validation_metrics(
    depth_map: np.ndarray,
    s: float,
    t: float,
    holdout_gcps: list[GCPAnchor],
) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """
    Compute out-of-sample Pearson r, MAE, RMSE against holdout GCPs.
    Returns (None, None, None) if there are fewer than _MIN_VALIDATION_POINTS.
    """
    if len(holdout_gcps) < _MIN_VALIDATION_POINTS:
        log.warning(
            "Only %d holdout GCPs — cannot compute meaningful validation metrics "
            "(need ≥ %d). Returning null metrics.",
            len(holdout_gcps),
            _MIN_VALIDATION_POINTS,
        )
        return None, None, None

    d_ho, z_ho = _sample_depth(depth_map, holdout_gcps)
    z_pred = s * d_ho + t

    r = _pearson_r(z_ho, z_pred)
    mae = float(np.mean(np.abs(z_pred - z_ho)))
    rmse = float(np.sqrt(np.mean((z_pred - z_ho) ** 2)))

    log.debug(
        "Holdout metrics (%d pts): r=%.3f  MAE=%.2f m  RMSE=%.2f m",
        len(holdout_gcps), r, mae, rmse,
    )
    return r, mae, rmse


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def calibrate(
    depth_map: np.ndarray,
    train_gcps: list[GCPAnchor],
    holdout_gcps: list[GCPAnchor],
) -> CalibrationResult:
    """
    Run the full dual-mode calibration pipeline.

    Args:
        depth_map    : HxW float32 normalised depth in [0, 1]
                       (1 = highest surface / closest to sensor).
        train_gcps   : GCPs used for RANSAC affine fitting (split='train').
        holdout_gcps : GCPs used only for validation metrics (split='holdout').

    Returns:
        :class:`CalibrationResult` containing the calibrated height matrix,
        affine parameters, inversion flag, fit method, and validation metrics.

    Raises:
        ValueError: If depth_map is not 2-D float32, or train_gcps is empty.
    """
    if depth_map.ndim != 2:
        raise ValueError(f"depth_map must be 2-D, got shape {depth_map.shape}")
    if not train_gcps:
        raise ValueError("train_gcps cannot be empty — need at least 1 train-split anchor.")

    depth = depth_map.copy().astype(np.float32)

    # ── Step 1: Sign check ──────────────────────────────────────────────────
    d_tr, z_tr = _sample_depth(depth, train_gcps)
    r_raw = _pearson_r(d_tr, z_tr)

    depth_inverted = r_raw < 0
    if depth_inverted:
        depth = 1.0 - depth
        d_tr = 1.0 - d_tr          # mirror the sampled values too
        log.info(
            "Sign check: r=%.3f < 0 — depth INVERTED (d̂ ← 1 − d̂) to correct "
            "camera-depth vs elevation-above-ground sign flip.",
            r_raw,
        )
    else:
        log.info("Sign check: r=%.3f ≥ 0 — depth orientation correct, no flip.", r_raw)

    # ── Step 2: Affine fit (RANSAC with OLS fallback) ───────────────────────
    if len(d_tr) < _MIN_RANSAC_POINTS:
        log.warning(
            "Only %d train GCPs — insufficient for RANSAC (need %d). Using OLS.",
            len(d_tr), _MIN_RANSAC_POINTS,
        )
        s_raw, t = _ols_affine(d_tr, z_tr)
        s = max(s_raw, 1e-6)
        fit_method = "ols"
    else:
        s, t, fit_method = _ransac_affine(d_tr, z_tr)

    log.info("Affine calibration: Z = %.4f·d̂ + %.4f  [%s]", s, t, fit_method)

    # ── Step 3: Apply affine to full depth map → metric height matrix ────────
    height_matrix = (s * depth + t).astype(np.float32)

    # ── Step 4: Validation metrics on holdout GCPs ──────────────────────────
    r_val, mae, rmse = _validation_metrics(depth, s, t, holdout_gcps)

    return CalibrationResult(
        depth_map=depth,
        affine_s=s,
        affine_t=t,
        depth_inverted=depth_inverted,
        fit_method=fit_method,
        metric_r=round(r_val, 4) if r_val is not None else None,
        metric_mae=round(mae, 4) if mae is not None else None,
        metric_rmse=round(rmse, 4) if rmse is not None else None,
        height_matrix=height_matrix,
    )
