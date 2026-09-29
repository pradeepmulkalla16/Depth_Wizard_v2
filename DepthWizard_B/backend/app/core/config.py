"""
backend/app/core/config.py
──────────────────────────
Centralised settings for DepthWizard backend.

All values can be overridden via environment variables (or a .env file).
Defaults assume a fresh local checkout with this layout:

  backend/
    app/
      core/config.py          ← this file
      data/
        models/da_v2_small.onnx
        presets/
        depthwizard.db
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

# ── optional .env loader ───────────────────────────────────────────────────
try:
    from dotenv import load_dotenv  # type: ignore

    _env_file = Path(__file__).resolve().parents[4] / ".env"
    if _env_file.exists():
        load_dotenv(_env_file)
except ImportError:
    pass  # python-dotenv not installed — rely on real env vars


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

# …/DepthWizard_B/backend/app/  (this file lives at .../core/config.py)
_APP_DIR: Path = Path(__file__).resolve().parent.parent
_DATA_DIR: Path = _APP_DIR / "data"


def _env_path(env_var: str, default: Path) -> Path:
    raw = os.getenv(env_var, "").strip()
    return Path(raw) if raw else default


def _env_str(env_var: str, default: str) -> str:
    return os.getenv(env_var, default).strip() or default


def _env_int(env_var: str, default: int) -> int:
    try:
        return int(os.getenv(env_var, str(default)))
    except (TypeError, ValueError):
        return default


def _cors_origins() -> List[str]:
    raw = os.getenv("CORS_ORIGINS", "").strip()
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()]
    return [
        "http://localhost:5173",   # Vite default
        "http://127.0.0.1:5173",
        "http://localhost:3000",   # CRA / alternative dev server
        "http://127.0.0.1:3000",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ]


# ---------------------------------------------------------------------------
# Settings — module-level constants (no Pydantic dependency here)
# ---------------------------------------------------------------------------

class Settings:
    """
    Immutable runtime settings loaded once at import time.
    Import the singleton:  ``from app.core.config import settings``
    """

    # ── Model ────────────────────────────────────────────────────────────────
    MODEL_PATH: Path = _env_path(
        "MODEL_PATH",
        _DATA_DIR / "models" / "da_v2_small.onnx",
    )

    # ── Root data directory (shared across services) ──────────────────────────
    DATA_DIR: Path = _DATA_DIR
    EXPORT_DIR: Path = _DATA_DIR / "exports"

    # ── Preset data ──────────────────────────────────────────────────────────
    PRESET_DATA_DIR: Path = _env_path(
        "PRESET_DATA_DIR",
        _DATA_DIR / "presets",
    )

    # ── Database ─────────────────────────────────────────────────────────────
    DB_PATH: Path = _env_path(
        "DB_PATH",
        _DATA_DIR / "depthwizard.db",
    )

    # ── Upload / inference ───────────────────────────────────────────────────
    MAX_UPLOAD_MB: int = _env_int("MAX_UPLOAD_MB", 20)
    INFERENCE_DEVICE: str = _env_str("INFERENCE_DEVICE", "cpu")

    # ── CORS ─────────────────────────────────────────────────────────────────
    # Evaluated at class definition time (after helpers are available).
    CORS_ORIGINS: List[str] = _cors_origins()
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: List[str] = ["*"]
    CORS_ALLOW_HEADERS: List[str] = ["*"]

    # ── API metadata ─────────────────────────────────────────────────────────
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_TITLE: str = "DepthWizard — Single-View DSM API"
    PROJECT_VERSION: str = "1.0.0"
    PROJECT_DESCRIPTION: str = (
        "Monocular depth → Digital Surface Model pipeline for ISRO SIH-26175."
    )


# Module-level singleton
settings = Settings()


# ---------------------------------------------------------------------------
# Startup helper — called from main.py lifespan event
# ---------------------------------------------------------------------------

def ensure_data_dirs() -> None:
    """Create all required runtime data directories (idempotent)."""
    for path in (
        settings.MODEL_PATH.parent,
        settings.PRESET_DATA_DIR,
        settings.DB_PATH.parent,
        settings.EXPORT_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def resolve_preset_asset_path(preset_id: str, stored_path: str | Path) -> Path:
    """Resolve a preset asset from its stored path or the bundled preset directory."""
    path = Path(stored_path)
    if path.is_file():
        return path
    return settings.PRESET_DATA_DIR / preset_id / path.name
