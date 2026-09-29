"""
backend/app/routers/reconstruct.py
────────────────────────────────────
POST /api/v1/reconstruct

Orchestrates the full DepthWizard AI pipeline:

  1. Load optical image (preset tile OR user upload).
  2. Run depth inference (Depth-Anything-V2 / synthetic dummy).
  3. Tactical mode  (preset_id with GCPs):
       a. Sign-check + RANSAC affine calibration against train-split GCPs.
       b. Compute validation metrics against holdout-split GCPs.
       c. Height matrix in metres AMSL.
  4. Relative mode (arbitrary upload / no calibration data):
       a. Normalise depth to [0, 100].
       b. calibration = null, metrics = null.
  5. Optional shadow-safe water leveling (VWI + Otsu + luminance gate).
  6. Downsample height matrix to 256×256 for fast Three.js rendering.
  7. Encode height matrix as base64 JSON payload.
  8. Log reconstruction metadata to SQLite.
  9. Return JSON matching PRD Section 10 contract.
"""

from __future__ import annotations

import base64
import io
import logging
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image as PilImage

from app.core.config import resolve_preset_asset_path, settings
from app.models.db import get_gcps, get_preset, insert_reconstruction
from app.services.calibrator import GCPAnchor, calibrate
from app.services.export import export_geotiff, export_glb
from app.services.inference import infer_depth, sharpen_urban_geometry
from app.services.water_leveling import level_water_body

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["reconstruction"])

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_OUTPUT_MESH_SIZE: int = 256    # downsampled resolution sent to Three.js
_MAX_UPLOAD_BYTES: int = settings.MAX_UPLOAD_MB * 1024 * 1024


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_preset_image(preset_id: str) -> tuple[np.ndarray, dict]:
    """
    Load the optical tile for a preset.

    Returns:
        (rgb_uint8, preset_row_as_dict)

    Raises:
        HTTPException 404 if preset not found.
        HTTPException 500 if the tile file is missing or unreadable.
    """
    row = get_preset(preset_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Preset '{preset_id}' not found.")

    tile_path = resolve_preset_asset_path(preset_id, row["tile_path"])
    if not tile_path.exists():
        raise HTTPException(
            status_code=500,
            detail=f"Preset tile missing on disk: {tile_path}",
        )

    try:
        img_bgr = cv2.imread(str(tile_path))
        if img_bgr is None:
            raise OSError("cv2.imread returned None")
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read preset tile: {exc}",
        ) from exc

    return rgb, dict(row)


def _load_upload_image(file: UploadFile) -> np.ndarray:
    """
    Decode an uploaded image file to uint8 RGB.

    Raises:
        HTTPException 400 for oversized or unreadable files.
    """
    raw = file.file.read()
    if len(raw) > _MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {settings.MAX_UPLOAD_MB} MB.",
        )
    try:
        pil_img = PilImage.open(io.BytesIO(raw)).convert("RGB")
        rgb = np.array(pil_img, dtype=np.uint8)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not decode image: {exc}",
        ) from exc
    return rgb


def _encode_height_matrix(height_matrix: np.ndarray) -> str:
    """
    Encode a float32 height matrix to a base64 string (raw bytes).

    Encoding:  float32 little-endian, shape prepended as two uint32 (H, W).
    Frontend decodes via DataView or Float32Array directly.
    """
    h, w = height_matrix.shape
    shape_bytes = np.array([h, w], dtype=np.uint32).tobytes()
    data_bytes = height_matrix.astype(np.float32).tobytes()
    return base64.b64encode(shape_bytes + data_bytes).decode("ascii")


def _downsample(matrix: np.ndarray, size: int = _OUTPUT_MESH_SIZE) -> np.ndarray:
    """Resize height matrix to `size × size` using area interpolation."""
    if matrix.shape == (size, size):
        return matrix.astype(np.float32)
    resized = cv2.resize(
        matrix.astype(np.float32),
        (size, size),
        interpolation=cv2.INTER_AREA,
    )
    return resized


def _gcps_from_rows(rows: list) -> list[GCPAnchor]:
    return [
        GCPAnchor(
            pixel_x=int(r["pixel_x"]),
            pixel_y=int(r["pixel_y"]),
            elevation_m=float(r["elevation_m"]),
            split=str(r["split"]),
        )
        for r in rows
    ]


def generate_depth_preview_base64(depth_array: np.ndarray) -> str:
    """Normalize depth array to 0-255 uint8, apply Inferno colormap, and return base64 PNG data URI."""
    norm_depth = depth_array.astype(np.float32)
    d_min, d_max = float(np.nanmin(norm_depth)), float(np.nanmax(norm_depth))
    if d_max > d_min:
        norm_depth = (norm_depth - d_min) / (d_max - d_min) * 255.0
    else:
        norm_depth = np.zeros_like(norm_depth)
    norm_depth = norm_depth.astype(np.uint8)

    # Apply Inferno colormap for technical diagnostics
    colored_depth = cv2.applyColorMap(norm_depth, cv2.COLORMAP_INFERNO)
    if colored_depth.shape[0] > 512:
        colored_depth = cv2.resize(colored_depth, (512, 512), interpolation=cv2.INTER_AREA)

    success, buffer = cv2.imencode('.png', colored_depth)
    if not success:
        return ""
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{b64_str}"


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

@router.post("/reconstruct", summary="Single-view DSM reconstruction")
async def reconstruct(
    preset_id:   Optional[str]        = Form(None,  description="Preset region ID (leh | hyderabad | assam)"),
    image_file:  Optional[UploadFile] = File(None,  description="Arbitrary image upload (JPG/PNG/TIFF)"),
    level_water: bool                 = Form(True,  description="Apply shadow-safe water body leveling"),
) -> JSONResponse:
    """
    Run the full DepthWizard reconstruction pipeline.

    Supply **either** `preset_id` (string) **or** `image_file` (binary upload),
    not both.  If both are supplied, `preset_id` takes precedence.

    Returns a JSON object matching the PRD Section 10 contract.
    """

    # ── Input validation ───────────────────────────────────────────────────
    if preset_id is None and image_file is None:
        raise HTTPException(
            status_code=422,
            detail="Provide either 'preset_id' or 'image_file'.",
        )

    rec_id = str(uuid.uuid4())
    preset_row: Optional[dict] = None

    # ── Step 1: Load image ─────────────────────────────────────────────────
    if preset_id:
        rgb_image, preset_row = _load_preset_image(preset_id)
        source_image_path = str(preset_row["tile_path"])
        log.info("[%s] Preset '%s' loaded — image %s", rec_id, preset_id, rgb_image.shape)
    else:
        rgb_image = _load_upload_image(image_file)                   # type: ignore[arg-type]
        source_image_path = f"upload::{image_file.filename}"
        log.info("[%s] Upload '%s' decoded — image %s", rec_id, image_file.filename, rgb_image.shape)

    # ── Step 2: Depth inference ────────────────────────────────────────────
    depth_map = infer_depth(rgb_image)   # HxW float32 [0,1]
    depth_preview_b64 = generate_depth_preview_base64(depth_map)
    log.info("[%s] Depth inference done — shape %s", rec_id, depth_map.shape)

    # ── Step 3 / 4: Calibration or relative normalisation ─────────────────
    mode: str
    height_matrix: np.ndarray
    units: str
    ground_resolution: Optional[float] = None
    calibration_payload: Optional[dict] = None
    metrics_payload: Optional[dict] = None
    affine_s: Optional[float] = None
    affine_t: Optional[float] = None
    depth_inverted: bool = False

    if preset_row is not None:
        # — TACTICAL MODE —
        all_gcps_rows = get_gcps(preset_id)                          # type: ignore[arg-type]
        train_rows    = [r for r in all_gcps_rows if r["split"] == "train"]
        holdout_rows  = [r for r in all_gcps_rows if r["split"] == "holdout"]

        if train_rows:
            train_gcps   = _gcps_from_rows(train_rows)
            holdout_gcps = _gcps_from_rows(holdout_rows)

            cal = calibrate(depth_map, train_gcps, holdout_gcps)

            height_matrix  = cal.height_matrix
            depth_inverted = cal.depth_inverted
            affine_s       = cal.affine_s
            affine_t       = cal.affine_t

            # Outlier shadow inversion clamping (especially for rugged high-relief alpine presets like Leh)
            if preset_id == "leh":
                p2, p98 = np.percentile(height_matrix, [2, 98])
                height_matrix = np.clip(height_matrix, p2, p98)

            calibration_payload = {
                "s":              round(cal.affine_s, 6),
                "t":              round(cal.affine_t, 6),
                "depth_inverted": cal.depth_inverted,
                "fit_method":     cal.fit_method,
            }
            metrics_payload = (
                {
                    "r":    round(cal.metric_r,    4) if cal.metric_r    is not None else None,
                    "mae":  round(cal.metric_mae,  4) if cal.metric_mae  is not None else None,
                    "rmse": round(cal.metric_rmse, 4) if cal.metric_rmse is not None else None,
                }
                if any(v is not None for v in (cal.metric_r, cal.metric_mae, cal.metric_rmse))
                else None
            )

            ground_resolution = float(preset_row["ground_resolution_m_per_px"])
            mode  = "tactical"
            units = "meters_amsl"
            log.info(
                "[%s] Tactical calibration: s=%.4f t=%.4f inverted=%s method=%s "
                "r=%s MAE=%s RMSE=%s",
                rec_id, cal.affine_s, cal.affine_t, cal.depth_inverted,
                cal.fit_method, cal.metric_r, cal.metric_mae, cal.metric_rmse,
            )
        else:
            # Preset exists but has no GCPs (shouldn't happen post-build_presets, but safe)
            log.warning(
                "[%s] Preset '%s' has no train GCPs — falling back to relative mode.",
                rec_id, preset_id,
            )
            height_matrix = (depth_map * 100.0).astype(np.float32)
            mode  = "relative"
            units = "relative_0_100"

    else:
        # — RELATIVE RECONNAISSANCE MODE —
        height_matrix = (depth_map * 100.0).astype(np.float32)
        mode  = "relative"
        units = "relative_0_100"
        log.info("[%s] Relative mode — depth normalised to [0, 100].", rec_id)

    # ── Step 5: Optional water leveling ────────────────────────────────────
    water_applied = False
    if level_water:
        height_matrix, water_applied = level_water_body(height_matrix, rgb_image)
        log.info("[%s] Water leveling: applied=%s", rec_id, water_applied)

    # ── Step 5.5: Sharpen structural edges & flat rooftops (edge-preserving filter)
    height_matrix = sharpen_urban_geometry(height_matrix, rgb_image)

    # ── Step 6: Downsample to 256×256 for Three.js ─────────────────────────
    height_256 = _downsample(height_matrix, _OUTPUT_MESH_SIZE)
    log.info("[%s] Height matrix downsampled to %s", rec_id, height_256.shape)

    # ── Step 7: Encode ─────────────────────────────────────────────────────
    encoded = _encode_height_matrix(height_256)

    # ── Step 8: Log to SQLite ──────────────────────────────────────────────
    try:
        insert_reconstruction(
            rec_id=rec_id,
            preset_id=preset_id,
            mode=mode,
            source_image_path=source_image_path,
            height_matrix_path=None,   # Phase 2: write .npy and store path
            affine_s=affine_s,
            affine_t=affine_t,
            depth_inverted=depth_inverted,
            ground_resolution_m_per_px=ground_resolution,
            metric_r=metrics_payload["r"]    if metrics_payload else None,
            metric_mae=metrics_payload["mae"]  if metrics_payload else None,
            metric_rmse=metrics_payload["rmse"] if metrics_payload else None,
        )
        log.info("[%s] Reconstruction logged to DB.", rec_id)
    except Exception as db_exc:
        log.warning("[%s] DB log failed (non-fatal): %s", rec_id, db_exc)

    # ── Step 9: Response ───────────────────────────────────────────────────
    return JSONResponse(content={
        "reconstruction_id":          rec_id,
        "mode":                       mode,
        "preset_id":                  preset_id,
        "height_matrix":              encoded,
        "height_matrix_shape":        list(height_256.shape),
        "units":                      units,
        "ground_resolution_m_per_px": ground_resolution,
        "calibration":                calibration_payload,
        "metrics":                    metrics_payload,
        "water_mask_applied":         water_applied,
        "depth_preview":              depth_preview_b64,
    })


# ---------------------------------------------------------------------------
# Export Endpoints (PRD Section 9.8)
# ---------------------------------------------------------------------------

def _decode_raw_height_matrix(b64_str: str) -> np.ndarray:
    """Decode base64 height matrix to 2D float32 numpy array."""
    raw = base64.b64decode(b64_str.strip().encode("ascii"))
    h, w = np.frombuffer(raw[:8], dtype=np.uint32)
    data = np.frombuffer(raw[8:8 + int(h * w * 4)], dtype=np.float32)
    return data.reshape((int(h), int(w)))


@router.post("/export/geotiff", summary="Export height matrix as 32-bit float GeoTIFF")
async def export_geotiff_route(
    preset_id: Optional[str] = Form(None),
    height_matrix: str = Form(..., description="Base64 encoded float32 height matrix"),
) -> FileResponse:
    """Generates a 32-bit float georeferenced GeoTIFF with Deflate compression."""
    import json
    try:
        matrix = _decode_raw_height_matrix(height_matrix)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid height matrix payload: {exc}") from exc

    bbox = [77.50, 34.10, 77.60, 34.20]  # default Leh AOI
    if preset_id:
        row = get_preset(preset_id)
        if row and row["bbox"]:
            try:
                bbox = json.loads(row["bbox"])
            except Exception:
                pass

    export_dir = settings.EXPORT_DIR
    export_dir.mkdir(parents=True, exist_ok=True)
    out_file = export_dir / f"depthwizard_{preset_id or 'surface'}_{uuid.uuid4().hex[:8]}.tif"

    try:
        export_geotiff(
            height_matrix=matrix,
            metadata={"bbox": bbox, "crs": "EPSG:4326"},
            output_path=out_file,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"GeoTIFF export failed: {exc}") from exc

    return FileResponse(
        path=str(out_file),
        filename=f"depthwizard_{preset_id or 'dsm'}.tif",
        media_type="image/tiff",
    )


@router.post("/export/glb", summary="Export height matrix with draped optical texture as GLB")
async def export_glb_route(
    preset_id: Optional[str] = Form(None),
    height_matrix: str = Form(..., description="Base64 encoded float32 height matrix"),
    vertical_exaggeration: float = Form(1.0),
) -> FileResponse:
    """Generates a binary 3D GLB triangular mesh with draped optical texture."""
    try:
        matrix = _decode_raw_height_matrix(height_matrix)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid height matrix payload: {exc}") from exc

    gsd = 10.0
    if preset_id:
        row = get_preset(preset_id)
        if row:
            gsd = float(row["ground_resolution_m_per_px"])
            rgb, _ = _load_preset_image(preset_id)
        else:
            rgb = np.full((matrix.shape[0], matrix.shape[1], 3), 180, dtype=np.uint8)
    else:
        rgb = np.full((matrix.shape[0], matrix.shape[1], 3), 180, dtype=np.uint8)

    export_dir = settings.EXPORT_DIR
    export_dir.mkdir(parents=True, exist_ok=True)
    out_file = export_dir / f"depthwizard_{preset_id or 'mesh'}_{uuid.uuid4().hex[:8]}.glb"

    try:
        export_glb(
            height_matrix=matrix,
            texture_image=rgb,
            metadata={"ground_resolution_m_per_px": gsd},
            output_path=out_file,
            z_scale=vertical_exaggeration,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"GLB export failed: {exc}") from exc

    return FileResponse(
        path=str(out_file),
        filename=f"depthwizard_{preset_id or 'terrain'}.glb",
        media_type="model/gltf-binary",
    )

