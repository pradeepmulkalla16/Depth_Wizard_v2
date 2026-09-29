"""
backend/app/services/export.py
───────────────────────────────
Export helpers for DepthWizard (PRD Section 9.8).

  export_geotiff  — writes metric height matrix as a georeferenced
                    32-bit float GeoTIFF via rasterio.
  export_glb      — writes an indexed triangular mesh with draped
                    texture as binary .glb via trimesh.

Both functions are gracefully guarded: if the optional dependency
(rasterio / trimesh) is not installed they raise ImportError with a
clear install hint rather than a cryptic AttributeError.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GeoTIFF export
# ---------------------------------------------------------------------------

def export_geotiff(
    height_matrix: np.ndarray,
    metadata: dict[str, Any],
    output_path: Path | str,
) -> Path:
    """
    Write a georeferenced 32-bit float GeoTIFF of the height matrix.

    Args:
        height_matrix : HxW float32 array — metric AMSL (or relative [0,100]).
        metadata      : dict that MUST contain:
                          bbox  — [min_lon, min_lat, max_lon, max_lat]
                          and optionally:
                          crs   — EPSG string, defaults to 'EPSG:4326'
                          nodata — nodata value, defaults to -9999.0
        output_path   : Destination .tif / .tiff path.

    Returns:
        Resolved Path of the written file.

    Raises:
        ImportError  : if rasterio is not installed.
        ValueError   : if height_matrix is not 2-D, or bbox is missing.
    """
    try:
        import rasterio                             # type: ignore
        from rasterio.transform import from_bounds  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "rasterio is required for GeoTIFF export. "
            "Install it with: pip install rasterio"
        ) from exc

    if height_matrix.ndim != 2:
        raise ValueError(
            f"height_matrix must be 2-D, got shape {height_matrix.shape}"
        )

    bbox = metadata.get("bbox")
    if bbox is None or len(bbox) != 4:
        raise ValueError(
            "metadata must contain 'bbox': [min_lon, min_lat, max_lon, max_lat]"
        )

    min_lon, min_lat, max_lon, max_lat = bbox
    h, w = height_matrix.shape
    crs = metadata.get("crs", "EPSG:4326")
    nodata = float(metadata.get("nodata", -9999.0))

    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, w, h)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = np.where(np.isfinite(height_matrix), height_matrix, nodata).astype(np.float32)

    with rasterio.open(
        str(out_path),
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype=rasterio.float32,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="deflate",
        predictor=3,      # floating-point predictor for better compression
        zlevel=6,
    ) as dst:
        dst.write(data, 1)

    size_kb = out_path.stat().st_size / 1024
    log.info("GeoTIFF exported → %s  (%.1f KB)", out_path, size_kb)
    return out_path


# ---------------------------------------------------------------------------
# GLB / mesh export
# ---------------------------------------------------------------------------

def _build_mesh_geometry(
    height_matrix: np.ndarray,
    gsd: float,
    z_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build vertices, faces, and UV coordinates for a height-field mesh.

    The mesh is axis-aligned in the XY plane:
      • X increases left → right  (pixel column)
      • Y increases bottom → top  (pixel row, Y-flipped for OpenGL convention)
      • Z is the displaced height

    Args:
        height_matrix : HxW float32
        gsd           : ground sampling distance in metres/pixel — used to
                        scale the XY footprint to real-world dimensions.
        z_scale       : additional vertical exaggeration factor (default 1.0).

    Returns:
        (vertices, faces, uvs)
        vertices : (N, 3) float32 — XYZ in metres
        faces    : (M, 3) int32   — triangle vertex indices
        uvs      : (N, 2) float32 — UV in [0,1]
    """
    h, w = height_matrix.shape

    # --- Build vertex grid -------------------------------------------------
    rows, cols = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")

    x = cols.astype(np.float32) * gsd                      # metres east
    y = (h - 1 - rows).astype(np.float32) * gsd            # metres north (Y-flipped)
    z = height_matrix.astype(np.float32) * z_scale

    # UV: u=col/(w-1), v=row/(h-1) in [0,1] — texture V is NOT flipped
    # (trimesh handles texture orientation).
    u = cols.astype(np.float32) / max(w - 1, 1)
    v = rows.astype(np.float32) / max(h - 1, 1)

    vertices = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)   # (H*W, 3)
    uvs      = np.stack([u.ravel(), v.ravel()],             axis=1)   # (H*W, 2)

    # --- Build triangle faces from quads -----------------------------------
    # Vertex index at (row, col): row*w + col
    i = rows[:-1, :-1].ravel() * w + cols[:-1, :-1].ravel()  # top-left
    j = rows[:-1, 1: ].ravel() * w + cols[:-1, 1: ].ravel()  # top-right
    k = rows[1:,  :-1].ravel() * w + cols[1:,  :-1].ravel()  # bottom-left
    l = rows[1:,  1: ].ravel() * w + cols[1:,  1: ].ravel()  # bottom-right

    tri_upper = np.stack([i, j, k], axis=1)   # upper triangle of quad
    tri_lower = np.stack([j, l, k], axis=1)   # lower triangle of quad
    faces = np.concatenate([tri_upper, tri_lower], axis=0).astype(np.int32)

    return vertices, faces, uvs


def export_glb(
    height_matrix: np.ndarray,
    texture_image: Optional[np.ndarray] = None,
    calibration: Any = None,
    metadata: Optional[dict[str, Any]] = None,
    output_path: Optional[Path | str] = None,
    gsd: Optional[float] = None,
    z_scale: float = 1.0,
    **kwargs: Any,
) -> Path:
    """
    Export the height-field mesh with a draped optical texture as binary GLB.

    Args:
        height_matrix : HxW float32 height values.
        texture_image : HxWx3 uint8 RGB optical image (used as mesh texture).
                        Will be resized to match height_matrix dimensions if needed.
        calibration   : Optional calibration telemetry dict.
        metadata      : Optional metadata dict (e.g. ground_resolution_m_per_px).
        output_path   : Destination .glb path.
        gsd           : metres/pixel — scales the XY mesh footprint (default 1.0 or from metadata).
        z_scale       : Vertical exaggeration multiplier (default 1.0).
        **kwargs      : Accepts any extra parameters safely without throwing TypeError.

    Returns:
        Resolved Path of the written .glb file.

    Raises:
        ImportError  : if trimesh is not installed.
        ValueError   : if height_matrix is not 2-D.
    """
    try:
        import trimesh                          # type: ignore
        import trimesh.visual.texture as tv    # type: ignore
        from PIL import Image as PilImage      # type: ignore
    except ImportError as exc:
        raise ImportError(
            "trimesh and Pillow are required for GLB export. "
            "Install with: pip install trimesh pillow"
        ) from exc

    if height_matrix.ndim != 2:
        raise ValueError(
            f"height_matrix must be 2-D, got shape {height_matrix.shape}"
        )

    import cv2  # already a hard dep

    # Resolve gsd from metadata or kwargs if not directly provided
    if gsd is None:
        if metadata and isinstance(metadata, dict):
            gsd = float(metadata.get("ground_resolution_m_per_px", metadata.get("gsd", 1.0)))
        elif "gsd" in kwargs:
            gsd = float(kwargs["gsd"])
        else:
            gsd = 1.0

    # Resolve output_path if passed via kwargs
    if output_path is None:
        if "out_path" in kwargs:
            output_path = kwargs["out_path"]
        elif "output_path" in kwargs:
            output_path = kwargs["output_path"]
        else:
            from app.core.config import settings
            import uuid
            output_path = settings.EXPORT_DIR / f"depthwizard_{uuid.uuid4().hex[:8]}.glb"

    h_hm, w_hm = height_matrix.shape

    # Default fallback texture if not provided
    if texture_image is None:
        texture_image = np.full((h_hm, w_hm, 3), 180, dtype=np.uint8)

    # Resize texture to match height matrix if needed
    h_tx, w_tx = texture_image.shape[:2]
    if (h_hm, w_hm) != (h_tx, w_tx):
        texture_image = cv2.resize(
            texture_image, (w_hm, h_hm), interpolation=cv2.INTER_LINEAR
        )

    # Build geometry
    vertices, faces, uvs = _build_mesh_geometry(height_matrix, gsd, z_scale)

    # Build trimesh TextureVisuals
    pil_texture = PilImage.fromarray(texture_image.astype(np.uint8))
    material = tv.SimpleMaterial(image=pil_texture)
    visual = tv.TextureVisuals(uv=uvs, material=material)

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        visual=visual,
        process=False,   # skip expensive mesh repair — our grid is already clean
        metadata=metadata if isinstance(metadata, dict) else None,
    )

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Export to GLB (binary GLTF)
    glb_bytes = mesh.export(file_type="glb")
    out_path.write_bytes(glb_bytes)

    size_kb = len(glb_bytes) / 1024
    log.info(
        "GLB exported → %s  (%.1f KB, %d vertices, %d faces)",
        out_path, size_kb, len(vertices), len(faces),
    )
    return out_path
