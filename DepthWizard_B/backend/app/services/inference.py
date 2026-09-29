"""
backend/app/services/inference.py
──────────────────────────────────
Depth-Anything-V2-Small ONNX inference wrapper.

Design guarantees
─────────────────
Guard 1 — Missing model  → synthetic dummy mode (never crashes on startup).
Guard 2 — ViT patch size → input always resized to (518, 518) before ONNX
           forward pass; disparity map resized back to original H×W with
           cv2.INTER_CUBIC.
Guard 3 — Edge snapping  → cv2.ximgproc.guidedFilter using optical luminance
           as guidance; falls back to Joint Bilateral Filter if ximgproc is
           unavailable.

Public API
──────────
    result = infer_depth(rgb_image: np.ndarray) -> np.ndarray
        • rgb_image : HxWx3  uint8  (0-255)
        • returns   : HxW    float32 normalised to [0, 1]
          where 1.0 = highest surface / closest to camera
          and   0.0 = lowest surface / furthest from camera
          (consistent with "relative elevation" convention used by calibrator)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import sys
from typing import Optional
import urllib.request

import cv2
import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "models"))
MODEL_PATH = os.path.join(MODEL_DIR, "da_v2_small.onnx")

# Reliable mirror for Depth Anything V2 Small ONNX (~95 MB)
MODEL_URL = "https://huggingface.co/onnx-community/Depth-Anything-V2-Small/resolve/main/onnx/model.onnx"


def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100.0, downloaded * 100.0 / total_size)
        sys.stdout.write(
            f"\r[DepthWizard] Downloading da_v2_small.onnx: {percent:.1f}% "
            f"({downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)"
        )
        sys.stdout.flush()
    else:
        sys.stdout.write(f"\r[DepthWizard] Downloading da_v2_small.onnx: {downloaded / (1024*1024):.1f} MB")
        sys.stdout.flush()


def ensure_model_weights(target_path: Optional[str | Path] = None) -> str:
    path = str(target_path) if target_path else MODEL_PATH
    model_dir = os.path.dirname(path)
    os.makedirs(model_dir, exist_ok=True)
    # Check if file is missing or corrupted/incomplete (less than 10 MB)
    if not os.path.exists(path) or os.path.getsize(path) < 10 * 1024 * 1024:
        print(f"[DepthWizard] ONNX model weights not found at {path}.")
        print(f"[DepthWizard] Fetching Depth Anything V2 Small weights from Hugging Face...")
        try:
            urllib.request.urlretrieve(MODEL_URL, path, reporthook=_reporthook)
            print("\n[DepthWizard] Model weights successfully downloaded and verified.")
        except Exception as e:
            print(f"\n[DepthWizard ERROR] Failed to download model weights: {e}")
            raise e
    return path


_VIT_INPUT_SIZE: int = 518          # Depth-Anything-V2 ViT 14-patch constraint
_PERCENTILE_LO: float = 2.0
_PERCENTILE_HI: float = 98.0

# Guided Filter hyper-params (empirically tuned for DSM edge sharpness)
_GF_RADIUS: int = 4
_GF_EPS: float = 0.01             # regularisation — lower = sharper edges

# Joint Bilateral fallback params
_JBF_D: int = 9
_JBF_SIGMA_COLOR: float = 75.0
_JBF_SIGMA_SPACE: float = 75.0


# ---------------------------------------------------------------------------
# ONNX session (lazy singleton)
# ---------------------------------------------------------------------------

class _OnnxSession:
    """Lazy-loaded, process-level ONNX Runtime session."""

    _session = None          # onnxruntime.InferenceSession or None
    _dummy_mode: bool = False

    @classmethod
    def load(cls, model_path: Path) -> None:
        """
        Attempt to load the ONNX model once.
        Sets _dummy_mode=True if the file is missing or loading fails.
        """
        try:
            resolved_path = ensure_model_weights(model_path)
            model_path = Path(resolved_path)
        except Exception as exc:
            log.warning(
                "⚠  Model download failed (%s) — checking existing file or dummy fallback.", exc
            )

        if not model_path.exists():
            log.warning(
                "⚠  Model NOT found at %s — running in SYNTHETIC DUMMY mode. "
                "Depth output will be procedurally generated. "
                "Download da_v2_small.onnx and set MODEL_PATH to enable real inference.",
                model_path,
            )
            cls._dummy_mode = True
            return

        try:
            import onnxruntime as ort  # type: ignore

            providers = ["CPUExecutionProvider"]
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 4
            opts.intra_op_num_threads = 4
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            cls._session = ort.InferenceSession(
                str(model_path),
                sess_options=opts,
                providers=providers,
            )
            cls._dummy_mode = False
            log.info("✓ Depth-Anything-V2-Small loaded from %s", model_path)

        except Exception as exc:
            log.warning(
                "⚠  ONNX load failed (%s) — falling back to SYNTHETIC DUMMY mode.", exc
            )
            cls._dummy_mode = True

    @classmethod
    def is_ready(cls) -> bool:
        return cls._session is not None

    @classmethod
    def run(cls, input_array: np.ndarray) -> np.ndarray:
        """Forward pass — input must be float32 NCHW (1,3,518,518)."""
        assert cls._session is not None
        input_name = cls._session.get_inputs()[0].name
        outputs = cls._session.run(None, {input_name: input_array})
        # Depth-Anything-V2 single output: (1, H, W) disparity
        return outputs[0].squeeze()  # → (H, W) float32


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _preprocess(rgb: np.ndarray) -> np.ndarray:
    """
    Resize to (518, 518), normalise to ImageNet stats, return NCHW float32.
    Guard 2 compliance: always resize to _VIT_INPUT_SIZE.
    """
    img = cv2.resize(rgb, (_VIT_INPUT_SIZE, _VIT_INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
    img = img.transpose(2, 0, 1)[np.newaxis, ...]  # HWC → NCHW
    return np.ascontiguousarray(img)


def _percentile_stretch(depth: np.ndarray) -> np.ndarray:
    """
    Contrast-stretch depth to [0, 1] using 2%–98% percentile clipping.
    Produces d̂ where 1 = highest/closest surface.
    """
    lo = float(np.percentile(depth, _PERCENTILE_LO))
    hi = float(np.percentile(depth, _PERCENTILE_HI))
    if hi - lo < 1e-6:
        return np.zeros_like(depth, dtype=np.float32)
    stretched = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
    return stretched.astype(np.float32)


# ---------------------------------------------------------------------------
# Guard 3 — Edge snapping (Guided Filter / Joint Bilateral fallback)
# ---------------------------------------------------------------------------

def _edge_snap(depth_f32: np.ndarray, rgb_guide: np.ndarray) -> np.ndarray:
    """
    Apply guided-filter edge snapping using optical luminance as guide image.

    Snaps rounded "pillowy" depth transitions to the true structural edges
    visible in the optical image.

    Args:
        depth_f32 : HxW float32 in [0, 1]
        rgb_guide : HxW×3 uint8 original optical image (same spatial dims)

    Returns:
        HxW float32 edge-snapped depth in [0, 1]
    """
    # Build single-channel luminance guide (float32, 0-1)
    luminance = cv2.cvtColor(rgb_guide, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    try:
        # Primary: ximgproc guided filter
        depth_uint8 = (depth_f32 * 255).astype(np.uint8)
        guide_uint8 = (luminance * 255).astype(np.uint8)

        filtered = cv2.ximgproc.guidedFilter(
            guide=guide_uint8,
            src=depth_uint8,
            radius=_GF_RADIUS,
            eps=(_GF_EPS * 255 ** 2),  # scale eps to uint8 domain
        )
        snapped = filtered.astype(np.float32) / 255.0
        log.debug("Edge snapping: guidedFilter applied (radius=%d, eps=%.4f)", _GF_RADIUS, _GF_EPS)

    except (AttributeError, cv2.error) as exc:
        # Fallback: Joint Bilateral Filter
        log.debug("guidedFilter unavailable (%s) — using Joint Bilateral Filter fallback.", exc)
        src_8u = (depth_f32 * 255).astype(np.uint8)
        guide_8u = (luminance * 255).astype(np.uint8)
        filtered = cv2.ximgproc.jointBilateralFilter(
            joint=guide_8u,
            src=src_8u,
            d=_JBF_D,
            sigmaColor=_JBF_SIGMA_COLOR,
            sigmaSpace=_JBF_SIGMA_SPACE,
        ) if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "jointBilateralFilter") else (
            cv2.bilateralFilter(src_8u, _JBF_D, _JBF_SIGMA_COLOR, _JBF_SIGMA_SPACE)
        )
        snapped = filtered.astype(np.float32) / 255.0

    # Re-normalise after filtering to guarantee output stays in [0, 1]
    return _percentile_stretch(snapped)


# ---------------------------------------------------------------------------
# Synthetic dummy depth (Guard 1 fallback)
# ---------------------------------------------------------------------------

def _synthetic_depth(rgb: np.ndarray) -> np.ndarray:
    """
    Generate a plausible-looking synthetic depth map when the ONNX model is
    unavailable.  Uses luminance gradients + mild Gaussian noise to produce a
    coherent "terrain-like" surface that exercises the full downstream pipeline.
    """
    h, w = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    # Smooth luminance (simulate long-range depth correlation)
    smoothed = cv2.GaussianBlur(gray, (0, 0), sigmaX=min(h, w) * 0.1)

    # Add structured low-frequency noise for terrain variety
    rng = np.random.default_rng(seed=42)
    noise_small = rng.standard_normal((h, w)).astype(np.float32) * 0.03
    noise_large = cv2.resize(
        rng.standard_normal((h // 8, w // 8)).astype(np.float32),
        (w, h),
        interpolation=cv2.INTER_LINEAR,
    ) * 0.08

    synthetic = smoothed + noise_small + noise_large
    stretched = _percentile_stretch(synthetic)
    log.debug("Synthetic dummy depth generated for %dx%d image.", w, h)
    return stretched


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def initialise(model_path: Optional[Path] = None) -> None:
    """
    Load the ONNX model (or enter dummy mode).
    Should be called once at application startup from the lifespan handler.
    """
    from app.core.config import settings  # avoid circular import at module load

    target = model_path or settings.MODEL_PATH
    _OnnxSession.load(target)


def infer_depth(rgb_image: np.ndarray) -> np.ndarray:
    """
    Run the full depth-inference + edge-snapping pipeline.

    Args:
        rgb_image: HxWx3 uint8 RGB image (not BGR).

    Returns:
        HxW float32 array in [0, 1].
        Value 1.0 → highest point on the visible surface.
        Value 0.0 → lowest point.

    Never raises — falls back to synthetic depth on any error.
    """
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError(
            f"rgb_image must be HxWx3, got shape {rgb_image.shape}"
        )

    orig_h, orig_w = rgb_image.shape[:2]

    # ── ONNX inference (Guard 1 + 2) ───────────────────────────────────────
    if _OnnxSession.is_ready():
        try:
            inp = _preprocess(rgb_image)                         # → NCHW (1,3,518,518)
            disparity = _OnnxSession.run(inp)                    # → (518, 518) float32

            # Guard 2: resize disparity back to original spatial dims
            disparity_orig = cv2.resize(
                disparity,
                (orig_w, orig_h),
                interpolation=cv2.INTER_CUBIC,
            )

            depth_norm = _percentile_stretch(disparity_orig)     # → [0, 1]

        except Exception as exc:
            log.warning("ONNX inference failed (%s) — falling back to synthetic depth.", exc)
            depth_norm = _synthetic_depth(rgb_image)
    else:
        depth_norm = _synthetic_depth(rgb_image)

    # ── Guard 3: Guided-Filter edge snapping ────────────────────────────────
    snapped = _edge_snap(depth_norm, rgb_image)
    sharpened = sharpen_urban_geometry(snapped, rgb_image)

    return sharpened


def sharpen_urban_geometry(depth_map: np.ndarray, rgb_image: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Separates fused building footprints by cutting down shadow alleys,
    flattening rooftop plateaus, and sharpening vertical perimeter walls.
    """
    # 1. Normalize depth to [0.0, 1.0] where 1.0 is roof, 0.0 is ground
    d_min, d_max = float(np.nanmin(depth_map)), float(np.nanmax(depth_map))
    if d_max > d_min:
        norm_h = (depth_map - d_min) / (d_max - d_min)
    else:
        return depth_map

    # Fallback if no rgb_image provided
    if rgb_image is None:
        filtered_depth = cv2.bilateralFilter(norm_h.astype(np.float32), d=7, sigmaColor=0.10, sigmaSpace=5.0)
        return (filtered_depth * (d_max - d_min) + d_min).astype(np.float32)

    # Ensure RGB matches depth map dimensions
    if rgb_image.shape[:2] != norm_h.shape:
        rgb_resized = cv2.resize(rgb_image, (norm_h.shape[1], norm_h.shape[0]), interpolation=cv2.INTER_LINEAR)
    else:
        rgb_resized = rgb_image

    gray = cv2.cvtColor(rgb_resized, cv2.COLOR_RGB2GRAY)

    # 2. Identify Shadow Alleys & Narrow Corridors Between Buildings
    # Deep shadows between towers have low luminance but get falsely elevated by depth networks
    shadow_mask = cv2.inRange(gray, 0, 65)  # Dark shadow pixels

    # Clean noise: remove tiny speckles, retain actual alleyways and courtyards
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    shadow_alleys = cv2.morphologyEx(shadow_mask, cv2.MORPH_OPEN, kernel_small)

    # 3. Carve Separation Channels (Drop Shadow Alleys to Street Level)
    # Wherever there is a shadow corridor between elevated structures, pull height down
    alley_indices = shadow_alleys > 0
    # Blend down alley height toward local ground baseline
    norm_h[alley_indices] = np.minimum(norm_h[alley_indices], 0.15)

    # 4. Morphological Edge Sharpening & Rooftop Plateau Flattening
    # Bilateral filter: d=7, sigmaColor=18 (tight colour tolerance for hard step edges),
    # sigmaSpace=4 (small spatial radius keeps walls vertical not smeared).
    sharp_h = cv2.bilateralFilter(norm_h.astype(np.float32), d=7, sigmaColor=18.0 / 255.0, sigmaSpace=4.0)

    # Gradient mask to accentuate wall cutoffs
    sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_magnitude = cv2.magnitude(sobel_x, sobel_y)
    edge_mask = cv2.normalize(edge_magnitude, None, 0, 1, cv2.NORM_MINMAX) > 0.35

    # Sharpen step transition along high-contrast building edges
    sharp_h[edge_mask] = np.where(gray[edge_mask] < 90, sharp_h[edge_mask] * 0.4, sharp_h[edge_mask] * 1.05)
    sharp_h = np.clip(sharp_h, 0.0, 1.0)

    # 5. Elevation Unsharp Mask: push high-frequency wall gradients steeper
    #    without moving flat rooftops or flat ground.
    from scipy.ndimage import gaussian_filter as _gf
    low_freq = _gf(sharp_h, sigma=2.0)
    high_freq = sharp_h - low_freq
    sharp_h = np.clip(sharp_h + 1.1 * high_freq, 0.0, 1.0)

    # 6. Restore original absolute/relative scale
    return (sharp_h * (d_max - d_min) + d_min).astype(np.float32)


# ---------------------------------------------------------------------------
# Public helper — bilateral + elevation-unsharp-mask for precomputed DSM arrays
# ---------------------------------------------------------------------------

def sharpen_dsm_edges(
    elev_matrix: np.ndarray,
    d: int = 7,
    sigma_color: float = 25.0,
    sigma_space: float = 5.0,
    unsharp_strength: float = 0.8,
) -> np.ndarray:
    """
    Post-process a precomputed metric DSM array with:
      1. Bilateral filter — smooths flat rooftops/ground while preserving step walls.
      2. Elevation Unsharp Mask — pushes edge gradients steeper without spiking.

    Designed for static preset generation (e.g. sync_preset_elevations.py).

    Args:
        elev_matrix    : 2-D float32 metric elevation array (any unit / scale).
        d              : Bilateral filter diameter (default 7).
        sigma_color    : Bilateral range sigma in raw elevation units (default 25 m).
        sigma_space    : Bilateral spatial sigma in pixels (default 5).
        unsharp_strength: Weight of the high-frequency elevation boost (default 0.8).

    Returns:
        float32 array with same shape and value range as elev_matrix.
    """
    from scipy.ndimage import gaussian_filter as _gf

    elev_f32 = elev_matrix.astype(np.float32)
    e_min = float(elev_f32.min())
    e_max = float(elev_f32.max())

    # 1. Bilateral filter on raw elevation values
    bilateral = cv2.bilateralFilter(elev_f32, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)

    # 2. Elevation Unsharp Mask: boosts high-frequency edge gradients
    low_freq = _gf(bilateral, sigma=2.0)
    high_freq = bilateral - low_freq
    sharpened = bilateral + unsharp_strength * high_freq

    # Clip to physical bounds to prevent inversion spikes
    return np.clip(sharpened, e_min, e_max).astype(np.float32)


@property
def dummy_mode() -> bool:
    """True if the service is running without a real ONNX model."""
    return _OnnxSession._dummy_mode
