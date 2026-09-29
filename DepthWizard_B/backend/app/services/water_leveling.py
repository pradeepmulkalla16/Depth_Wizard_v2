"""
backend/app/services/water_leveling.py
────────────────────────────────────────
Shadow-safe water body leveling (PRD Section 9.3).

Pipeline
────────
1. Compute RGB Visible Water Index (VWI):
       VWI = (Green − Red) / (Green + Red + 1e-5)
2. Otsu thresholding on the VWI map → water candidate mask.
3. Luminance gate: reject pixels where luminance < τ_shadow to prevent
   dark terrain shadows (mountain ravines, building shadows) from being
   misclassified as water and incorrectly flattened.
4. For each connected water region, clamp height to the minimum baseline
   elevation within that region.

Public API
──────────
    leveled, applied = level_water_body(
        height_matrix : np.ndarray,   # HxW float32
        rgb_image     : np.ndarray,   # HxWx3 uint8
        tau_shadow    : float = 0.25, # luminance gate threshold (0-1)
    ) -> tuple[np.ndarray, bool]
"""

from __future__ import annotations

import logging
from typing import Tuple

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_TAU_SHADOW: float = 0.25    # pixels darker than this are shadows, not water
_MIN_WATER_PIXELS: int = 50          # ignore tiny isolated water detections (noise)
_MORPH_KERNEL_SIZE: int = 3          # morphological close to fill small holes in mask


# ---------------------------------------------------------------------------
# Step 1: Visible Water Index (3-band RGB proxy, no NIR required)
# ---------------------------------------------------------------------------

def _compute_vwi(rgb: np.ndarray) -> np.ndarray:
    """
    Compute per-pixel Visible Water Index using the 3-band RGB proxy.

    VWI = (Green − Red) / (Green + Red + 1e-5)

    Returns a float32 array in (−1, +1).
    High positive values indicate water-like spectral response.
    """
    rgb_f = rgb.astype(np.float32)
    green = rgb_f[..., 1]
    red   = rgb_f[..., 0]
    vwi   = (green - red) / (green + red + 1e-5)
    return vwi.astype(np.float32)


# ---------------------------------------------------------------------------
# Step 2: Otsu thresholding on VWI
# ---------------------------------------------------------------------------

def _otsu_water_mask(vwi: np.ndarray) -> np.ndarray:
    """
    Convert VWI to uint8 and apply Otsu thresholding.
    Returns a boolean mask (True = water candidate).
    """
    # Shift VWI from (−1,+1) → (0, 255) uint8 for Otsu
    vwi_u8 = np.clip((vwi + 1.0) * 127.5, 0, 255).astype(np.uint8)
    threshold, binary = cv2.threshold(
        vwi_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    log.debug("Otsu VWI threshold: %.1f (uint8 scale)", threshold)
    return binary.astype(bool)


# ---------------------------------------------------------------------------
# Step 3: Luminance gate
# ---------------------------------------------------------------------------

def _compute_luminance(rgb: np.ndarray) -> np.ndarray:
    """Return perceptual luminance in [0, 1] from uint8 RGB."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return gray.astype(np.float32) / 255.0


def _apply_luminance_gate(
    water_candidates: np.ndarray,
    luminance: np.ndarray,
    tau: float,
) -> np.ndarray:
    """
    Remove dark pixels from the water mask.

    Rationale: real water in nadir optical imagery is generally bright/blue
    (high luminance).  Dark shadows from mountains, ravines, or buildings can
    accidentally satisfy the VWI water criterion — the luminance gate rejects
    them before any height leveling occurs.

    Args:
        water_candidates : boolean mask from Otsu (True = water candidate)
        luminance        : float32 [0,1] per-pixel luminance
        tau              : pixels with luminance < tau are rejected as shadow

    Returns:
        Refined boolean mask (dark pixels excluded).
    """
    not_shadow = luminance >= tau
    refined = water_candidates & not_shadow
    n_before = int(water_candidates.sum())
    n_after  = int(refined.sum())
    log.debug(
        "Luminance gate (τ=%.2f): water pixels %d → %d (removed %d shadows)",
        tau, n_before, n_after, n_before - n_after,
    )
    return refined


# ---------------------------------------------------------------------------
# Step 4: Morphological clean-up + per-region baseline clamping
# ---------------------------------------------------------------------------

def _clean_mask(mask: np.ndarray) -> np.ndarray:
    """
    Morphological closing to fill small holes then erosion to remove
    isolated single-pixel detections.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (_MORPH_KERNEL_SIZE, _MORPH_KERNEL_SIZE)
    )
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    # Remove isolated specks smaller than _MIN_WATER_PIXELS
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    clean = np.zeros_like(closed, dtype=np.uint8)
    for label in range(1, n_labels):   # skip background (label 0)
        if stats[label, cv2.CC_STAT_AREA] >= _MIN_WATER_PIXELS:
            clean[labels == label] = 1
    return clean.astype(bool)


def _clamp_water_regions(
    height_matrix: np.ndarray,
    water_mask: np.ndarray,
) -> np.ndarray:
    """
    For each connected water region, clamp all pixels to the per-region
    minimum elevation (the "baseline" elevation of that water body).

    This removes artificial surface waviness caused by monocular depth
    uncertainty over specular or turbid water surfaces.
    """
    if not water_mask.any():
        return height_matrix

    h_out = height_matrix.copy()
    n_labels, labels = cv2.connectedComponents(water_mask.astype(np.uint8), connectivity=8)

    total_leveled = 0
    for label in range(1, n_labels):
        region = labels == label
        n_px = int(region.sum())
        if n_px < _MIN_WATER_PIXELS:
            continue
        baseline = float(np.min(height_matrix[region]))
        h_out[region] = baseline
        total_leveled += n_px
        log.debug(
            "Water region %d: %d px clamped to baseline %.2f",
            label, n_px, baseline,
        )

    log.info("Water leveling: %d px across %d region(s) clamped.", total_leveled, n_labels - 1)
    return h_out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def level_water_body(
    height_matrix: np.ndarray,
    rgb_image: np.ndarray,
    tau_shadow: float = _DEFAULT_TAU_SHADOW,
) -> Tuple[np.ndarray, bool]:
    """
    Apply shadow-safe water leveling to a height matrix.

    Args:
        height_matrix : HxW float32 height values (metric AMSL or relative [0,100]).
        rgb_image     : HxWx3 uint8 RGB optical image (same spatial dims).
        tau_shadow    : Luminance gate threshold; pixels darker than this are
                        classified as shadow and excluded from water leveling.
                        Range [0, 1]; default 0.25.

    Returns:
        (leveled_matrix, water_mask_applied)
        leveled_matrix    : HxW float32 — same as input but with water pixels
                            clamped to per-region baseline elevation.
        water_mask_applied: True if any water pixels were detected and leveled;
                            False if the water mask was empty (no leveling done).

    Never raises — any internal failure returns the unmodified height matrix
    with water_mask_applied=False.
    """
    try:
        if height_matrix.ndim != 2:
            raise ValueError(f"height_matrix must be 2-D, got shape {height_matrix.shape}")
        if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
            raise ValueError(f"rgb_image must be HxWx3, got shape {rgb_image.shape}")

        # Resize optical image to match height matrix if dims differ
        h_hm, w_hm = height_matrix.shape
        h_im, w_im = rgb_image.shape[:2]
        if (h_hm, w_hm) != (h_im, w_im):
            log.debug(
                "Resizing rgb_image from %dx%d to %dx%d to match height_matrix.",
                w_im, h_im, w_hm, h_hm,
            )
            rgb_image = cv2.resize(rgb_image, (w_hm, h_hm), interpolation=cv2.INTER_LINEAR)

        # Step 1: VWI
        vwi = _compute_vwi(rgb_image)

        # Step 2: Otsu
        water_candidates = _otsu_water_mask(vwi)

        # Step 3: Luminance gate
        luminance = _compute_luminance(rgb_image)
        water_refined = _apply_luminance_gate(water_candidates, luminance, tau_shadow)

        # Step 4: Morphological clean-up
        water_clean = _clean_mask(water_refined)

        if not water_clean.any():
            log.info("Water leveling: no water pixels detected — height matrix unchanged.")
            return height_matrix.copy(), False

        # Step 5: Per-region baseline clamping
        leveled = _clamp_water_regions(height_matrix, water_clean)
        return leveled, True

    except Exception as exc:
        log.warning(
            "Water leveling failed (%s) — returning unmodified height matrix.", exc
        )
        return height_matrix.copy(), False
