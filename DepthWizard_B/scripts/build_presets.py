#!/usr/bin/env python3
"""
scripts/build_presets.py
────────────────────────
Standalone preset-seeding script for DepthWizard.

What this does
──────────────
For each of the three bundled regions (leh, hyderabad, assam):

  1. Generates a 512×512 synthetic optical RGB PNG that visually resembles
     the described terrain type (mountain, urban, floodplain).
  2. Generates a matching 512×512 float32 GeoTIFF acting as the reference DEM
     with realistic absolute elevation ranges and spatial variance.
  3. Writes a metadata.json (bbox, GSD, sensor info).
  4. Extracts 30 GCP anchors uniformly across the DEM:
       • 15 → split='train'  (used for RANSAC affine fit)
       • 15 → split='holdout' (used ONLY for validation metrics — never fitting)
  5. Inserts presets + GCP anchors into depthwizard.db.

Usage
─────
  # From the DepthWizard_B directory, with the venv active:
  python scripts/build_presets.py

  # Rebuild from scratch (drops and recreates existing preset rows):
  python scripts/build_presets.py --reset

Environment
───────────
Reads MODEL_PATH / PRESET_DATA_DIR / DB_PATH from environment (or .env);
falls back to the same defaults as config.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Make sure the backend package is importable even when running this script
# directly (python scripts/build_presets.py) without an editable install.
_SCRIPT_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _SCRIPT_DIR.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import numpy as np

try:
    import rasterio  # type: ignore
    from rasterio.transform import from_bounds  # type: ignore
    _RASTERIO_OK = True
except ImportError:
    _RASTERIO_OK = False
    logging.warning("rasterio not installed — DEM files will be written as raw .npy instead of GeoTIFF.")

from PIL import Image  # type: ignore

from app.core.config import settings, ensure_data_dirs
from app.models.db import init_db, get_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Preset specifications
# ---------------------------------------------------------------------------

PRESET_SPECS: list[dict[str, Any]] = [
    {
        "id": "leh",
        "name": "Leh — Ladakh Highlands",
        "bbox": [77.35, 33.85, 77.75, 34.25],   # [min_lon, min_lat, max_lon, max_lat]
        "ground_resolution_m_per_px": 10.0,       # Sentinel-2 GSD
        "sensor": "Sentinel-2 L2A (Bands 4-3-2)",
        # Terrain model parameters
        "base_elevation_m": 3500.0,
        "elevation_range_m": 1800.0,              # high variance — rugged mountain
        "terrain_type": "mountain",
    },
    {
        "id": "hyderabad",
        "name": "Hyderabad — Deccan Plateau",
        "bbox": [78.30, 17.25, 78.70, 17.65],
        "ground_resolution_m_per_px": 10.0,
        "sensor": "Sentinel-2 L2A (Bands 4-3-2)",
        "base_elevation_m": 540.0,
        "elevation_range_m": 120.0,               # low variance — urban plateau
        "terrain_type": "urban_plateau",
    },
    {
        "id": "assam",
        "name": "Assam — Brahmaputra Floodplain",
        "bbox": [92.80, 26.00, 93.20, 26.40],
        "ground_resolution_m_per_px": 10.0,
        "sensor": "Sentinel-2 L2A (Bands 4-3-2)",
        "base_elevation_m": 120.0,
        "elevation_range_m": 40.0,                # very low variance — river plain
        "terrain_type": "floodplain",
    },
]

_IMAGE_SIZE: int = 512
_N_GCPS: int = 30
_TRAIN_COUNT: int = 15
_HOLDOUT_COUNT: int = 15


# ---------------------------------------------------------------------------
# Synthetic terrain DEM generators
# ---------------------------------------------------------------------------

def _make_perlin_like(
    shape: tuple[int, int],
    octaves: int = 6,
    persistence: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    """
    Build a normalised [0, 1] heightmap using a multi-octave noise stack.
    Pure NumPy — no external noise library required.
    """
    h, w = shape
    rng = np.random.default_rng(seed)
    result = np.zeros((h, w), dtype=np.float64)
    amplitude = 1.0
    max_amplitude = 0.0

    for octave in range(octaves):
        freq = 2 ** octave
        # Sample noise at coarse grid then upsample (cheap "Perlin" substitute)
        gh = max(2, h // freq)
        gw = max(2, w // freq)
        noise = rng.random((gh, gw))
        # Bicubic upsampling via PIL
        upsampled = np.array(
            Image.fromarray((noise * 255).astype(np.uint8)).resize(
                (w, h), Image.BICUBIC
            )
        ).astype(np.float64) / 255.0
        result += amplitude * upsampled
        max_amplitude += amplitude
        amplitude *= persistence

    return (result / max_amplitude).astype(np.float32)


def _generate_mountain_dem(size: int, base: float, elev_range: float, seed: int = 1) -> np.ndarray:
    """Rugged highland terrain: high-frequency ridges, sharp valleys."""
    base_noise = _make_perlin_like((size, size), octaves=7, persistence=0.6, seed=seed)
    # Add a dominant ridge running NW→SE
    ridge = np.zeros((size, size), dtype=np.float32)
    for i in range(size):
        peak = int(size * 0.35 + 0.3 * size * np.sin(i / size * np.pi))
        width = max(20, int(size * 0.15))
        col_range = np.arange(size)
        ridge[i] = np.exp(-0.5 * ((col_range - peak) / width) ** 2)

    dem = base_noise * 0.6 + ridge * 0.4
    # Normalise and scale
    dem = (dem - dem.min()) / (dem.max() - dem.min() + 1e-8)
    return (base + dem * elev_range).astype(np.float32)


def _generate_urban_plateau_dem(size: int, base: float, elev_range: float, seed: int = 2) -> np.ndarray:
    """Flat plateau with subtle building clusters and gentle undulation."""
    rng = np.random.default_rng(seed)
    # Very smooth base
    base_noise = _make_perlin_like((size, size), octaves=4, persistence=0.3, seed=seed)
    # Sparse building blobs
    buildings = np.zeros((size, size), dtype=np.float32)
    n_clusters = 15
    for _ in range(n_clusters):
        cx = rng.integers(30, size - 30)
        cy = rng.integers(30, size - 30)
        bh = float(rng.uniform(0.15, 0.50))   # relative height
        bw = int(rng.integers(10, 35))
        bh_px = int(rng.integers(10, 35))
        r0, r1 = max(0, cy - bh_px // 2), min(size, cy + bh_px // 2)
        c0, c1 = max(0, cx - bw // 2),  min(size, cx + bw // 2)
        buildings[r0:r1, c0:c1] = np.maximum(buildings[r0:r1, c0:c1], bh)

    dem = base_noise * 0.5 + buildings * 0.5
    dem = (dem - dem.min()) / (dem.max() - dem.min() + 1e-8)
    return (base + dem * elev_range).astype(np.float32)


def _generate_floodplain_dem(size: int, base: float, elev_range: float, seed: int = 3) -> np.ndarray:
    """Flat floodplain with a sinuous river channel and oxbow features."""
    rng = np.random.default_rng(seed)
    base_noise = _make_perlin_like((size, size), octaves=3, persistence=0.25, seed=seed)

    # River channel: sinusoidal path across the image
    river_mask = np.zeros((size, size), dtype=np.float32)
    river_depth = 0.4   # fraction of elev_range below plain
    channel_width = int(size * 0.05)

    for row in range(size):
        # Sinuous centre column
        cx = int(size * 0.5 + 0.2 * size * np.sin(row / size * 3 * np.pi))
        cx = int(np.clip(cx, channel_width, size - channel_width))
        col_range = np.arange(size)
        mask_row = np.exp(-0.5 * ((col_range - cx) / channel_width) ** 2)
        river_mask[row] = mask_row

    dem = base_noise * 0.6 - river_mask * river_depth
    dem = (dem - dem.min()) / (dem.max() - dem.min() + 1e-8)
    return (base + dem * elev_range).astype(np.float32)


def _generate_dem(spec: dict[str, Any]) -> np.ndarray:
    """Dispatch to the appropriate DEM generator by terrain_type."""
    terrain = spec["terrain_type"]
    size = _IMAGE_SIZE
    base = spec["base_elevation_m"]
    elev_range = spec["elevation_range_m"]
    seed_map = {"leh": 1, "hyderabad": 2, "assam": 3}
    seed = seed_map.get(spec["id"], 99)

    if terrain == "mountain":
        return _generate_mountain_dem(size, base, elev_range, seed)
    elif terrain == "urban_plateau":
        return _generate_urban_plateau_dem(size, base, elev_range, seed)
    elif terrain == "floodplain":
        return _generate_floodplain_dem(size, base, elev_range, seed)
    else:
        raise ValueError(f"Unknown terrain_type: {terrain!r}")


# ---------------------------------------------------------------------------
# Synthetic optical image generators
# ---------------------------------------------------------------------------

def _dem_to_optical_mountain(dem: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Rocky grey peaks + snow caps + shadow valleys."""
    norm = (dem - dem.min()) / (dem.max() - dem.min() + 1e-8)
    h, w = norm.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    # Rocky base (grey-brown)
    rgb[..., 0] = 0.55 + norm * 0.25   # R
    rgb[..., 1] = 0.50 + norm * 0.20   # G
    rgb[..., 2] = 0.40 + norm * 0.18   # B

    # Snow: pixels above 85th percentile elevation → white
    snow_thresh = float(np.percentile(norm, 85))
    snow = norm > snow_thresh
    rgb[snow, 0] = 0.92 + rng.random(snow.sum()) * 0.08
    rgb[snow, 1] = 0.92 + rng.random(snow.sum()) * 0.08
    rgb[snow, 2] = 0.95 + rng.random(snow.sum()) * 0.05

    # Shadow valleys (dark)
    shadow = norm < 0.15
    rgb[shadow] *= 0.4

    # Texture noise
    noise = rng.standard_normal((h, w, 3)).astype(np.float32) * 0.04
    rgb = np.clip(rgb + noise, 0.0, 1.0)
    return (rgb * 255).astype(np.uint8)


def _dem_to_optical_urban(dem: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Grey urban fabric with green parks and beige open land."""
    norm = (dem - dem.min()) / (dem.max() - dem.min() + 1e-8)
    h, w = norm.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    # Base: dry Deccan scrubland — brownish
    rgb[..., 0] = 0.60 + norm * 0.10
    rgb[..., 1] = 0.55 + norm * 0.08
    rgb[..., 2] = 0.38 + norm * 0.05

    # Urban built-up (high elevation = buildings) → grey
    built = norm > 0.50
    rgb[built, 0] = 0.55 + rng.random(built.sum()) * 0.1
    rgb[built, 1] = 0.55 + rng.random(built.sum()) * 0.1
    rgb[built, 2] = 0.55 + rng.random(built.sum()) * 0.1

    # Green parks (random patches at mid elevation)
    park_seed = rng.integers(0, h * w, size=int(h * w * 0.04))
    rows, cols = np.unravel_index(park_seed, (h, w))
    rgb[rows, cols, 0] = 0.25
    rgb[rows, cols, 1] = 0.50
    rgb[rows, cols, 2] = 0.25

    noise = rng.standard_normal((h, w, 3)).astype(np.float32) * 0.03
    rgb = np.clip(rgb + noise, 0.0, 1.0)
    return (rgb * 255).astype(np.uint8)


def _dem_to_optical_floodplain(dem: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Green agricultural patches + blue river channel."""
    norm = (dem - dem.min()) / (dem.max() - dem.min() + 1e-8)
    h, w = norm.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)

    # Agriculture — lush green
    rgb[..., 0] = 0.25 + norm * 0.20
    rgb[..., 1] = 0.52 + norm * 0.15
    rgb[..., 2] = 0.20 + norm * 0.10

    # River (very low elevation) → blue
    river = norm < 0.15
    rgb[river, 0] = 0.15 + rng.random(river.sum()) * 0.10
    rgb[river, 1] = 0.35 + rng.random(river.sum()) * 0.10
    rgb[river, 2] = 0.65 + rng.random(river.sum()) * 0.15

    # Sandbars (slightly elevated river margins) → beige
    sandbar = (norm > 0.14) & (norm < 0.22)
    rgb[sandbar, 0] = 0.80
    rgb[sandbar, 1] = 0.74
    rgb[sandbar, 2] = 0.55

    noise = rng.standard_normal((h, w, 3)).astype(np.float32) * 0.025
    rgb = np.clip(rgb + noise, 0.0, 1.0)
    return (rgb * 255).astype(np.uint8)


def _generate_optical(spec: dict[str, Any], dem: np.ndarray) -> np.ndarray:
    """Return 512×512 uint8 RGB matching the terrain type."""
    rng = np.random.default_rng({"leh": 10, "hyderabad": 20, "assam": 30}.get(spec["id"], 99))
    terrain = spec["terrain_type"]
    if terrain == "mountain":
        return _dem_to_optical_mountain(dem, rng)
    elif terrain == "urban_plateau":
        return _dem_to_optical_urban(dem, rng)
    elif terrain == "floodplain":
        return _dem_to_optical_floodplain(dem, rng)
    raise ValueError(f"Unknown terrain_type: {terrain!r}")


# ---------------------------------------------------------------------------
# GeoTIFF writer
# ---------------------------------------------------------------------------

def _write_dem_tif(dem: np.ndarray, tif_path: Path, spec: dict[str, Any]) -> None:
    """Write the DEM as a 32-bit float GeoTIFF with proper spatial reference."""
    bbox = spec["bbox"]
    min_lon, min_lat, max_lon, max_lat = bbox
    h, w = dem.shape
    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, w, h)

    with rasterio.open(
        str(tif_path),
        "w",
        driver="GTiff",
        height=h,
        width=w,
        count=1,
        dtype=rasterio.float32,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(dem.astype(np.float32), 1)


def _write_dem_npy(dem: np.ndarray, npy_path: Path) -> None:
    """Fallback when rasterio is unavailable — save as .npy."""
    np.save(str(npy_path), dem.astype(np.float32))


# ---------------------------------------------------------------------------
# GCP extraction
# ---------------------------------------------------------------------------

def _extract_gcps(
    dem: np.ndarray,
    n_total: int = _N_GCPS,
    train_count: int = _TRAIN_COUNT,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """
    Extract n_total uniformly distributed GCP anchors from the DEM.

    Returns:
        (train_gcps, holdout_gcps) — each is a list of dicts with keys
        pixel_x, pixel_y, elevation_m.
    """
    h, w = dem.shape
    rng = np.random.default_rng(seed)

    # Stratified uniform grid: n_total points spread across the image
    grid_rows = int(np.ceil(np.sqrt(n_total)))
    grid_cols = int(np.ceil(n_total / grid_rows))

    row_edges = np.linspace(0, h, grid_rows + 1, dtype=int)
    col_edges = np.linspace(0, w, grid_cols + 1, dtype=int)

    points: list[dict] = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            if len(points) >= n_total:
                break
            # Pick a random pixel within this grid cell
            py = int(rng.integers(row_edges[r], max(row_edges[r] + 1, row_edges[r + 1])))
            px = int(rng.integers(col_edges[c], max(col_edges[c] + 1, col_edges[c + 1])))
            py = int(np.clip(py, 0, h - 1))
            px = int(np.clip(px, 0, w - 1))
            elev = float(dem[py, px])
            points.append({"pixel_x": px, "pixel_y": py, "elevation_m": elev})

    # Shuffle and split
    rng.shuffle(points)  # type: ignore[arg-type]
    train = points[:train_count]
    holdout = points[train_count: train_count + _HOLDOUT_COUNT]
    return train, holdout


# ---------------------------------------------------------------------------
# Database seeding
# ---------------------------------------------------------------------------

def _upsert_preset(conn: sqlite3.Connection, spec: dict[str, Any]) -> None:
    tile_path = str(Path(spec["id"]) / "optical.png")
    dem_path = (
        str(Path(spec["id"]) / "reference_dem.tif")
        if _RASTERIO_OK
        else str(Path(spec["id"]) / "reference_dem.npy")
    )
    conn.execute(
        """
        INSERT OR REPLACE INTO presets
            (id, name, bbox, tile_path, dem_path, ground_resolution_m_per_px, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            spec["id"],
            spec["name"],
            json.dumps(spec["bbox"]),
            tile_path,
            dem_path,
            spec["ground_resolution_m_per_px"],
            datetime.now(timezone.utc).isoformat(),
        ),
    )


def _upsert_gcps(
    conn: sqlite3.Connection,
    preset_id: str,
    train_gcps: list[dict],
    holdout_gcps: list[dict],
    reset: bool,
) -> None:
    if reset:
        conn.execute("DELETE FROM gcp_anchors WHERE preset_id = ?", (preset_id,))

    for gcp in train_gcps:
        conn.execute(
            "INSERT INTO gcp_anchors (preset_id, pixel_x, pixel_y, elevation_m, split) "
            "VALUES (?, ?, ?, ?, 'train')",
            (preset_id, gcp["pixel_x"], gcp["pixel_y"], gcp["elevation_m"]),
        )
    for gcp in holdout_gcps:
        conn.execute(
            "INSERT INTO gcp_anchors (preset_id, pixel_x, pixel_y, elevation_m, split) "
            "VALUES (?, ?, ?, ?, 'holdout')",
            (preset_id, gcp["pixel_x"], gcp["pixel_y"], gcp["elevation_m"]),
        )


# ---------------------------------------------------------------------------
# Per-preset build pipeline
# ---------------------------------------------------------------------------

def build_preset(spec: dict[str, Any], reset: bool = False) -> None:
    preset_id = spec["id"]
    preset_dir: Path = settings.PRESET_DATA_DIR / preset_id
    preset_dir.mkdir(parents=True, exist_ok=True)

    log.info("━━━  Building preset: %s (%s)  ━━━", preset_id, spec["name"])

    # 1. Generate DEM
    log.info("  [1/5] Generating synthetic DEM …")
    dem = _generate_dem(spec)
    log.info(
        "        DEM stats: min=%.1fm  max=%.1fm  mean=%.1fm  std=%.1fm",
        dem.min(), dem.max(), dem.mean(), dem.std(),
    )

    # 2. Write DEM as GeoTIFF (or .npy fallback)
    log.info("  [2/5] Writing DEM to disk …")
    if _RASTERIO_OK:
        tif_path = preset_dir / "reference_dem.tif"
        _write_dem_tif(dem, tif_path, spec)
        log.info("        Saved → %s", tif_path)
    else:
        npy_path = preset_dir / "reference_dem.npy"
        _write_dem_npy(dem, npy_path)
        log.info("        Saved (npy fallback) → %s", npy_path)

    # 3. Generate optical image
    log.info("  [3/5] Generating synthetic optical RGB image …")
    optical = _generate_optical(spec, dem)
    optical_path = preset_dir / "optical.png"
    Image.fromarray(optical).save(str(optical_path))
    log.info("        Saved → %s", optical_path)

    # 4. Write metadata.json
    log.info("  [4/5] Writing metadata.json …")
    metadata = {
        "preset_id": preset_id,
        "name": spec["name"],
        "bbox": spec["bbox"],
        "gsd_m_per_px": spec["ground_resolution_m_per_px"],
        "sensor": spec["sensor"],
        "terrain_type": spec["terrain_type"],
        "dem_stats": {
            "min_m": float(dem.min()),
            "max_m": float(dem.max()),
            "mean_m": float(dem.mean()),
            "std_m": float(dem.std()),
        },
        "image_size_px": _IMAGE_SIZE,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = preset_dir / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    log.info("        Saved → %s", meta_path)

    # 5. Extract GCPs and seed database
    log.info("  [5/5] Extracting %d GCPs (%d train / %d holdout) and seeding DB …",
             _N_GCPS, _TRAIN_COUNT, _HOLDOUT_COUNT)
    seed_map = {"leh": 101, "hyderabad": 202, "assam": 303}
    train_gcps, holdout_gcps = _extract_gcps(dem, seed=seed_map.get(preset_id, 42))

    with get_connection() as conn:
        _upsert_preset(conn, spec)
        _upsert_gcps(conn, preset_id, train_gcps, holdout_gcps, reset=reset)

    log.info(
        "  ✓  Preset '%s' ready — %d train + %d holdout GCPs inserted.",
        preset_id, len(train_gcps), len(holdout_gcps),
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build / reseed DepthWizard preset tiles, DEMs, and GCP anchors."
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing GCP anchors for each preset before reinserting. "
             "Presets rows are always upserted (INSERT OR REPLACE).",
    )
    parser.add_argument(
        "--presets",
        nargs="*",
        default=None,
        help="Space-separated list of preset IDs to build (default: all).",
    )
    args = parser.parse_args()

    # Ensure all directories and schema exist
    ensure_data_dirs()
    init_db()

    target_ids = set(args.presets) if args.presets else {s["id"] for s in PRESET_SPECS}
    specs_to_build = [s for s in PRESET_SPECS if s["id"] in target_ids]

    if not specs_to_build:
        log.error("No matching presets found for: %s", args.presets)
        sys.exit(1)

    log.info("Building %d preset(s): %s", len(specs_to_build), [s["id"] for s in specs_to_build])

    for spec in specs_to_build:
        try:
            build_preset(spec, reset=args.reset)
        except Exception:
            log.exception("Failed to build preset '%s' — continuing with others.", spec["id"])

    log.info("═══  All presets built.  DB: %s  ═══", settings.DB_PATH)


if __name__ == "__main__":
    main()
