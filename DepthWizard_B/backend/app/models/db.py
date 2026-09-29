"""
backend/app/models/db.py
────────────────────────
SQLite schema initialisation for DepthWizard.

Tables (matching PRD Section 8):
  • presets          — bundled optical tile metadata + GSD
  • gcp_anchors      — ground-control points, split into train / holdout
  • reconstructions  — per-request reconstruction log (tactical + relative)
  • elevation_profiles — ruler measurements derived from a reconstruction

Usage:
    from app.models.db import init_db, get_connection
    init_db()                        # idempotent — safe to call on every startup
    with get_connection() as conn:
        conn.execute("SELECT ...")
"""

from __future__ import annotations

import sqlite3
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from app.core.config import settings

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DDL — tables are created with IF NOT EXISTS so init_db() is idempotent
# ---------------------------------------------------------------------------

_DDL_PRESETS = """
CREATE TABLE IF NOT EXISTS presets (
    id                          TEXT    PRIMARY KEY,
    name                        TEXT    NOT NULL,
    bbox                        TEXT    NOT NULL,           -- JSON [min_lon, min_lat, max_lon, max_lat]
    tile_path                   TEXT    NOT NULL,           -- relative path to optical.png
    dem_path                    TEXT,                       -- relative path to reference_dem.tif (nullable)
    ground_resolution_m_per_px  REAL    NOT NULL,           -- GSD in metres/pixel
    created_at                  DATETIME NOT NULL DEFAULT (datetime('now'))
);
"""

_DDL_GCP_ANCHORS = """
CREATE TABLE IF NOT EXISTS gcp_anchors (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    preset_id   TEXT    NOT NULL REFERENCES presets(id) ON DELETE CASCADE,
    pixel_x     INTEGER NOT NULL,
    pixel_y     INTEGER NOT NULL,
    elevation_m REAL    NOT NULL,                           -- ground-truth elevation AMSL
    split       TEXT    NOT NULL CHECK(split IN ('train','holdout'))
                                                            -- train → used for RANSAC fit
                                                            -- holdout → used only for validation metrics
);
"""

_DDL_RECONSTRUCTIONS = """
CREATE TABLE IF NOT EXISTS reconstructions (
    id                          TEXT    PRIMARY KEY,        -- UUID
    preset_id                   TEXT    REFERENCES presets(id),   -- NULL → arbitrary upload
    mode                        TEXT    NOT NULL CHECK(mode IN ('tactical','relative')),
    source_image_path           TEXT    NOT NULL,
    height_matrix_path          TEXT,                       -- path to .npy / encoded payload
    affine_s                    REAL,                       -- scale  (NULL in relative mode; enforced > 0)
    affine_t                    REAL,                       -- offset (NULL in relative mode)
    depth_inverted              BOOLEAN NOT NULL DEFAULT 0, -- 1 if raw depth was flipped before fitting
    ground_resolution_m_per_px  REAL,                       -- NULL in relative/uncalibrated mode
    metric_r                    REAL,                       -- Pearson r vs holdout GCPs (NULL in relative)
    metric_mae                  REAL,                       -- MAE  metres (NULL in relative)
    metric_rmse                 REAL,                       -- RMSE metres (NULL in relative)
    created_at                  DATETIME NOT NULL DEFAULT (datetime('now'))
);
"""

_DDL_ELEVATION_PROFILES = """
CREATE TABLE IF NOT EXISTS elevation_profiles (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    reconstruction_id   TEXT    NOT NULL REFERENCES reconstructions(id) ON DELETE CASCADE,
    point_a_x           INTEGER NOT NULL,
    point_a_y           INTEGER NOT NULL,
    point_b_x           INTEGER NOT NULL,
    point_b_y           INTEGER NOT NULL,
    distance_m          REAL,                               -- NULL if ground_resolution_m_per_px absent
    elevation_delta_m   REAL,
    slope_percent       REAL
);
"""

# Indices for common query patterns
_DDL_INDICES = [
    "CREATE INDEX IF NOT EXISTS idx_gcp_preset ON gcp_anchors(preset_id, split);",
    "CREATE INDEX IF NOT EXISTS idx_recon_preset ON reconstructions(preset_id);",
    "CREATE INDEX IF NOT EXISTS idx_profile_recon ON elevation_profiles(reconstruction_id);",
]

_ALL_DDL: list[str] = [
    _DDL_PRESETS,
    _DDL_GCP_ANCHORS,
    _DDL_RECONSTRUCTIONS,
    _DDL_ELEVATION_PROFILES,
    *_DDL_INDICES,
]


# ---------------------------------------------------------------------------
# Connection factory
# ---------------------------------------------------------------------------

def _db_path() -> Path:
    return settings.DB_PATH


@contextmanager
def get_connection() -> Generator[sqlite3.Connection, None, None]:
    """
    Yield a sqlite3 connection with row_factory=sqlite3.Row (dict-like access)
    and WAL journal mode for better concurrent read performance.

    Usage::

        with get_connection() as conn:
            rows = conn.execute("SELECT * FROM presets").fetchall()
    """
    conn = sqlite3.connect(str(_db_path()))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------

def init_db() -> None:
    """
    Create all tables and indices (idempotent — IF NOT EXISTS guards).
    Safe to call on every application startup.
    """
    # Ensure the parent directory exists before trying to open the DB file.
    _db_path().parent.mkdir(parents=True, exist_ok=True)

    with get_connection() as conn:
        for statement in _ALL_DDL:
            try:
                conn.execute(statement)
            except sqlite3.Error as exc:
                log.error("DDL failed: %s\n  SQL: %s", exc, statement.strip()[:120])
                raise

    log.info("Database initialised at: %s", _db_path())


# ---------------------------------------------------------------------------
# Convenience CRUD helpers (thin wrappers — routers may use these directly)
# ---------------------------------------------------------------------------

def get_preset(preset_id: str) -> sqlite3.Row | None:
    """Return a single preset row or None."""
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM presets WHERE id = ?", (preset_id,)
        ).fetchone()


def list_presets() -> list[sqlite3.Row]:
    """Return all preset rows."""
    with get_connection() as conn:
        return conn.execute("SELECT * FROM presets ORDER BY name").fetchall()


def get_gcps(preset_id: str, split: str | None = None) -> list[sqlite3.Row]:
    """
    Return GCP anchors for a preset.

    Args:
        preset_id: The preset identifier.
        split:     If 'train' or 'holdout', filter by that split;
                   if None, return all anchors.
    """
    if split is not None:
        sql = "SELECT * FROM gcp_anchors WHERE preset_id = ? AND split = ?"
        params: tuple = (preset_id, split)
    else:
        sql = "SELECT * FROM gcp_anchors WHERE preset_id = ?"
        params = (preset_id,)

    with get_connection() as conn:
        return conn.execute(sql, params).fetchall()


def insert_reconstruction(
    *,
    rec_id: str,
    preset_id: str | None,
    mode: str,
    source_image_path: str,
    height_matrix_path: str | None,
    affine_s: float | None,
    affine_t: float | None,
    depth_inverted: bool,
    ground_resolution_m_per_px: float | None,
    metric_r: float | None,
    metric_mae: float | None,
    metric_rmse: float | None,
) -> None:
    """Insert a new reconstruction log row."""
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO reconstructions (
                id, preset_id, mode, source_image_path, height_matrix_path,
                affine_s, affine_t, depth_inverted, ground_resolution_m_per_px,
                metric_r, metric_mae, metric_rmse
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                rec_id, preset_id, mode, source_image_path, height_matrix_path,
                affine_s, affine_t, int(depth_inverted),
                ground_resolution_m_per_px, metric_r, metric_mae, metric_rmse,
            ),
        )


def insert_elevation_profile(
    *,
    reconstruction_id: str,
    point_a_x: int,
    point_a_y: int,
    point_b_x: int,
    point_b_y: int,
    distance_m: float | None,
    elevation_delta_m: float | None,
    slope_percent: float | None,
) -> int:
    """Insert an elevation-profile measurement and return its rowid."""
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO elevation_profiles (
                reconstruction_id,
                point_a_x, point_a_y, point_b_x, point_b_y,
                distance_m, elevation_delta_m, slope_percent
            ) VALUES (?,?,?,?,?,?,?,?)
            """,
            (
                reconstruction_id,
                point_a_x, point_a_y, point_b_x, point_b_y,
                distance_m, elevation_delta_m, slope_percent,
            ),
        )
        return cur.lastrowid  # type: ignore[return-value]
