"""
backend/app/main.py
────────────────────
FastAPI application entry point for DepthWizard.

Responsibilities
─────────────────
• Create the FastAPI app with lifespan context manager.
• Register CORS middleware (origins from config.py).
• On startup: ensure data directories exist, initialise SQLite schema,
  and warm-load the ONNX inference session (or enter dummy mode).
• Mount /static/presets → backend/app/data/presets/ for tile thumbnails.
• Include API routers:
    - /api/v1/reconstruct   (POST)
    - /api/v1/presets       (GET, GET /{id})
• Health-check endpoint: GET /health
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import ensure_data_dirs, settings
from app.models.db import init_db
from app.routers import presets as presets_router
from app.routers import reconstruct as reconstruct_router
from app.services import inference as inference_svc

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Logging setup (only configure if no handlers already registered by the
# runtime — avoids double-formatting when uvicorn owns the root logger)
# ---------------------------------------------------------------------------

_handler = logging.StreamHandler()
_handler.setFormatter(
    logging.Formatter("%(asctime)s  %(levelname)-8s  [%(name)s]  %(message)s")
)
root_log = logging.getLogger()
if not root_log.handlers:
    root_log.addHandler(_handler)
root_log.setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Lifespan — startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler.

    Startup sequence:
      1. Create data directories (models/, presets/, db parent).
      2. Initialise SQLite schema (idempotent).
      3. Load ONNX inference session (or enter synthetic-dummy mode).
    """
    log.info("━━━  DepthWizard backend starting  ━━━")

    # 1. Data dirs
    ensure_data_dirs()
    log.info("Data directories verified.")

    # 2. DB schema
    init_db()

    # 3. Model weights verification / auto-download
    try:
        inference_svc.ensure_model_weights()
    except Exception as exc:
        log.warning("⚠ Model weights check/download encountered an issue: %s", exc)

    # 4. Inference session
    inference_svc.initialise()

    log.info(
        "━━━  Startup complete — inference=%s  ━━━",
        "ONNX" if inference_svc._OnnxSession.is_ready() else "SYNTHETIC_DUMMY",
    )

    yield   # app is live here

    log.info("━━━  DepthWizard backend shutting down  ━━━")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.PROJECT_TITLE,
        version=settings.PROJECT_VERSION,
        description=settings.PROJECT_DESCRIPTION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── CORS ─────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
        allow_methods=settings.CORS_ALLOW_METHODS,
        allow_headers=settings.CORS_ALLOW_HEADERS,
    )

    # ── Static files: preset tiles / thumbnails ───────────────────────────
    presets_dir: Path = settings.PRESET_DATA_DIR
    if presets_dir.exists():
        app.mount(
            "/static/presets",
            StaticFiles(directory=str(presets_dir)),
            name="presets_static",
        )
        log.info("Static preset files mounted at /static/presets → %s", presets_dir)
    else:
        log.warning(
            "Preset data directory not found (%s) — /static/presets not mounted. "
            "Run scripts/build_presets.py to generate preset assets.",
            presets_dir,
        )

    # ── Routers ───────────────────────────────────────────────────────────
    app.include_router(reconstruct_router.router)
    app.include_router(presets_router.router)

    # ── Health check ──────────────────────────────────────────────────────
    @app.get("/health", tags=["meta"], summary="Health check")
    def health() -> dict:
        return {
            "status": "ok",
            "version": settings.PROJECT_VERSION,
            "inference_mode": (
                "onnx" if inference_svc._OnnxSession.is_ready() else "synthetic_dummy"
            ),
        }

    return app


# Module-level singleton used by uvicorn
app = create_app()


# ---------------------------------------------------------------------------
# Dev runner  (python -m app.main  or  python backend/app/main.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
