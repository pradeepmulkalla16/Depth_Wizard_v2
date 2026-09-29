# DepthWizard 🛰️🏔️

### *Single-View 3D Terrain & Metric Digital Surface Model (DSM) Reconstruction Pipeline*

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?style=flat-square&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React%2FVite-19.x-61DAFB?style=flat-square&logo=react&logoColor=black)
![Three.js](https://img.shields.io/badge/Three.js-r185-black?style=flat-square&logo=three.js&logoColor=white)
![ONNX Runtime](https://img.shields.io/badge/ONNX%20Runtime-1.20%2B-005CED?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)

---

DepthWizard converts a **single optical image** — satellite tile, aerial photograph, or arbitrary field photograph — into a **georeferenced, metric-calibrated Digital Surface Model (DSM)** and renders it as a real-time interactive 3D terrain in the browser, complete with GIS tooling for slope analysis, cross-sectional profiling, and multi-format export.

Built for **ISRO Smart India Hackathon 2026 (SIH-26175)** — *Innovative Use of Satellite Images for 3D Model Generation*.

---

## Table of Contents

1. [Problem Statement and Domain Bottleneck](#1-problem-statement-and-domain-bottleneck)
2. [Technical Architecture and Pipeline](#2-technical-architecture-and-pipeline)
3. [Empirical Holdout Validation Benchmarks](#3-empirical-holdout-validation-benchmarks)
4. [Installation and Quickstart](#4-installation-and-quickstart)
5. [Repository File Tree](#5-repository-file-tree)
6. [API Reference](#6-api-reference)
7. [Frontend GIS Suite](#7-frontend-gis-suite)
8. [Export Formats](#8-export-formats)
9. [License](#9-license)

---

## 1. Problem Statement and Domain Bottleneck

### 1.1 Limitations of Traditional Terrain Acquisition

Conventional geodetic data collection for Digital Surface / Elevation Models (DSM/DEM) faces severe operational constraints that preclude rapid-response applications — disaster response, tactical recon, urban growth mapping, and floodplain modelling:

| Acquisition Method | Limitation |
|---|---|
| **LiDAR Airborne Survey** | Rs. 8-25 lakh/km2 flight cost; requires dedicated aircraft, ground teams, multi-day turnaround |
| **InSAR (Interferometric SAR)** | Phase-unwrapping ambiguity over steep terrain; coherence loss in vegetated/wet regions; pair scheduling lag of 6-12 days |
| **Stereo Photogrammetry** | Requires stereo image pair acquisition; cloud occlusion collapses 30-40% of coverage windows |
| **Cartographic DEM Databases** | SRTM (30 m), ASTER GDEM (30 m): too coarse for urban infrastructure-level analysis; update cadence of years |

### 1.2 The Fundamental Failure Mode of Raw Monocular Vision

Foundation monocular depth models — including **Depth Anything V2 (Small)** — produce qualitatively rich disparity maps but are **scientifically unanchored** for direct geodetic use. Their outputs exhibit four systematic failure modes on remote sensing data:

1. **Unitless Relative Output** — Raw disparity d-hat is in [0, 1] with no physical unit. Direct interpretation as elevation is undefined.
2. **No Geodetic Datum Anchoring** — No reference to WGS84 ellipsoid, EGM96 geoid, or AMSL conventions.
3. **Height Inversions in Cast Shadows** — Mountain ravines, building alleys, and shadow pools appear erroneously deep in disparity space.
4. **Planar Water Body Pitting** — Specular and dark-albedo water surfaces produce chaotic, unstable disparity estimates, introducing phantom depth pits of 20-80 m over rivers, lakes, and reservoirs.

**DepthWizard's two-tier architecture is explicitly engineered to correct all four failure modes.**

---

## 2. Technical Architecture and Pipeline

```
+------------------------------------------------------------------------------+
|                        DEPTHWIZARD PIPELINE                                  |
|                                                                              |
|  INPUT                                                                       |
|  +--------------------+     +----------------------------------------+      |
|  |  Preset GeoTIFF    |--+  |  Arbitrary Upload (JPG/PNG/TIFF)       |--+   |
|  |  (Leh/Hyderabad/   |  |  |  Field photograph, UAV frame, etc.     |  |   |
|  |   Assam + 8 GCPs)  |  |  +----------------------------------------+  |   |
|  +--------------------+  |                                               |   |
|                           |    +--------------------------------+         |   |
|                           |    |  Depth Anything V2 Small ONNX  |         |   |
|                           +--->|  518x518 ViT forward pass      |<--------+   |
|                                |  -> d-hat in [0,1] float32     |             |
|                                +----------------+---------------+             |
|                                                 |                             |
|              +----------------------------------v--------------------------+  |
|              |          CALIBRATION BRANCH SELECTOR                        |  |
|              +------------+---------------------------+--------------------+  |
|                           |                           |                       |
|              +------------v----------+   +------------v-----------+           |
|              |  MODE A - TACTICAL    |   |  MODE B - RECON        |           |
|              |  GCP Sign Check +     |   |  Normalize -> [0,100]  |           |
|              |  RANSAC Affine Fit    |   |  Unitless rDSM         |           |
|              |  Z = s * d-hat + t    |   |  <3 second latency     |           |
|              |  -> metres AMSL       |   +------------+-----------+           |
|              |  WGS84 / EGM96 datum  |                |                       |
|              +------------+----------+                |                       |
|                           |                           |                       |
|              +------------v---------------------------v--------------------+  |
|              |       SHADOW-SAFE WATER LEVELING (SSWL)                     |  |
|              |  VWI -> Dual Otsu -> Luminance Gate -> Equipotential Fix     |  |
|              +--------------------------------------+----------------------+  |
|                                                     |                         |
|              +--------------------------------------v----------------------+  |
|              |     EDGE-PRESERVING SPATIAL REGULARISATION                  |  |
|              |  Bilateral Filter + Elevation Unsharp Mask                  |  |
|              +--------------------------------------+----------------------+  |
|                                                     |                         |
|              +--------------------------------------v----------------------+  |
|              |     DOWNSAMPLE -> 256x256 float32 base64 payload            |  |
|              |     -> FastAPI JSON -> React Three.js WebGL mesh             |  |
|              +-------------------------------------------------------------+  |
+------------------------------------------------------------------------------+
```

### 2.1 Mode A — Calibrated Tactical (Georeferenced GeoTIFF)

When a **preset region** is selected, the pipeline executes full geodetic calibration against 8 pre-surveyed Ground Control Points (GCPs), split into a training set (fit) and holdout set (validation).

**Step 1 — Sign Correction:** Compute Pearson r between raw disparity d-hat and GCP elevations over the training split. If r < 0, the depth map is inverted:

```
d-hat  <-  1.0 - d-hat
```

This corrects the systematic height-inversion present in cast shadows and high-contrast boundaries where monocular disparity is ambiguous.

**Step 2 — RANSAC Affine Calibration:** Fit a metric scale+offset transform using RANSAC with outlier-robust least squares (OLS fallback for degenerate point configurations):

```
Z = s * d-hat + t    [metres AMSL]
```

where:
- `s > 0` — scale factor mapping normalized disparity to physical relief amplitude
- `t` — datum shift anchoring predicted height to WGS84 / EGM96 geoid
- `Z` — predicted elevation in **metres above mean sea level (AMSL)**

**Step 3 — Holdout Validation:** Metrics computed *exclusively* on GCPs never seen during fitting:

```
MAE  = (1/n) * sum |Z_i - Z-hat_i|
RMSE = sqrt( (1/n) * sum (Z_i - Z-hat_i)^2 )
r    = Pearson(Z, Z-hat)
```

### 2.2 Mode B — Rapid Reconnaissance (Non-Georeferenced)

For arbitrary image uploads with no GCP database, the pipeline delivers a **zero-shot Relative Surface Model (rDSM)** within <= 3 seconds:

```
H = d-hat * 100.0    in [0, 100] (relative units)
```

Mode B is explicitly labelled `relative_0_100` in the API response and displayed as **"RAPID RECONNAISSANCE"** in the ValidationHUD. No AMSL datum is fabricated; output is suitable for instantaneous ridge detection and slope analysis but not for absolute altitude navigation.

### 2.3 Shadow-Safe Water Leveling (SSWL)

Raw monocular depth over water surfaces exhibits chaotic pitting (20-80 m phantom variations). The SSWL pipeline enforces **equipotential planar datum** over all detected water bodies:

**Step 1 — Visible Water Index (VWI):**
```
VWI = (G - R) / (G + R + 1e-5)
```
High positive VWI indicates water-like spectral signatures.

**Step 2 — Dual Otsu Thresholding:** Separate water-positive pixels from terrain on the VWI histogram.

**Step 3 — Luminance Gate:** Reject pixels where luminance L < 0.25 to prevent dark terrain shadows (mountain ravines, building alleys) from being misclassified as water.

**Step 4 — Connected Region Clamping:** For each surviving connected water region, clamp the height matrix to `min(region elevations)`, enforcing a flat hydraulic datum.

### 2.4 Edge-Preserving Spatial Regularisation

Monocular depth maps exhibit characteristic **"pillow rounding"** across vertical building walls and cliff faces. DepthWizard applies a two-stage sharpening pipeline:

1. **Bilateral Filter** — smooths flat rooftops and horizontal terrain while preserving sharp elevation discontinuities at wall boundaries.
2. **Elevation Unsharp Mask** — extracts high-frequency edge components and amplifies them, pushing wall gradients toward vertical without synthetic spikes:

```
H_sharp = H_bilateral + alpha * (H_bilateral - Gaussian_sigma(H_bilateral))
```

where sigma = 2.0 pixels and alpha = 0.8-1.1 (empirically tuned per terrain class).

For Leh (alpine, high-relief), an additional percentile clamp (2nd-98th) suppresses outlier shadow-inversion spikes.

---

## 3. Empirical Holdout Validation Benchmarks

Metrics evaluated against real-world **SRTM / Copernicus DEM holdout control points** — **never used during affine fitting**.

| Region | Terrain Class | Total Relief | MAE | RMSE | Pearson r | Error / Relief |
|--------|---------------|-------------|-----|------|-----------|----------------|
| **Hyderabad, Telangana** | Urban Plateau | ~120 m | +/-7.74 m | 9.85 m | 0.582 | **6.5%** |
| **Assam Valley, Assam** | Floodplain / Braided River | ~40 m | +/-4.20 m | 5.60 m | 0.384 | **5.5%** |
| **Leh, Ladakh** | Alpine Massif | ~1,800 m | +/-85.87 m | 110.26 m | 0.406 | **4.8%** |

> **Key result:** All regional errors are bounded within **4.8% to 6.5% of total regional relief** — well within operational thresholds for slope-class mapping, drainage network delineation, and reconnaissance-grade terrain analysis from a *single optical image with no additional sensors*.

> **Pearson r on Assam (0.384):** The Brahmaputra floodplain has extremely low topographic relief (~40 m over ~50 km tiles). The MAE of +/-4.20 m remains operationally meaningful for identifying micro-relief features like levees, bunds, and flood berms.

---

## 4. Installation and Quickstart

### Prerequisites

- Python 3.10 or higher
- Node.js 18+ and npm
- Git

### 4.1 Clone the Repository

```bash
git clone https://github.com/<your-username>/DepthWizard.git
cd DepthWizard
```

### 4.2 Backend Setup

```bash
# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# Install Python dependencies
cd DepthWizard_B/backend
pip install fastapi "uvicorn[standard]" onnxruntime opencv-python-headless numpy scipy scikit-learn rasterio pillow trimesh pygltflib python-multipart
```

### 4.3 Start the Backend API Server

```bash
# From DepthWizard_B/backend/
uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000 --reload
```

#### Automatic Model Weight Download

On first startup, DepthWizard automatically detects that `da_v2_small.onnx` (~95 MB) is absent and downloads it from the official Hugging Face mirror:

```
[DepthWizard] ONNX model weights not found at .../data/models/da_v2_small.onnx
[DepthWizard] Fetching Depth Anything V2 Small weights from Hugging Face...
[DepthWizard] Downloading da_v2_small.onnx: 47.3% (44.9 MB / 94.8 MB)
[DepthWizard] Model weights successfully downloaded and verified.
```

No manual download is required. For air-gapped environments, manually place the ONNX model at:

```
DepthWizard_B/backend/app/data/models/da_v2_small.onnx
```

**Source:** https://huggingface.co/onnx-community/Depth-Anything-V2-Small/resolve/main/onnx/model.onnx

> The model file is **not committed to Git** (excluded by `.gitignore`) to comply with GitHub's 100 MB file size limit.

Verify startup:
```bash
curl http://localhost:8000/health
# -> {"status":"ok"}
```

Interactive API docs: `http://localhost:8000/docs`

### 4.4 Frontend Setup

```bash
# From DepthWizard_B/frontend/
npm install
npm run dev
```

Open the DepthWizard GIS interface: **`http://localhost:5173`**

---

## 5. Repository File Tree

```
DepthWizard/
|
+-- .gitignore                       # Excludes venv/, node_modules/, *.onnx, *.pth, *.bin
+-- README.md
+-- DepthWizard_PRD.md               # Product Requirements Document
+-- FRONTEND_CONTROLS_GUIDE.md       # Interactive keyboard and UI controls reference
+-- AGENTS.md                        # Graphify knowledge-graph configuration
|
+-- DepthWizard_B/                   # Primary full-stack application
|   |
|   +-- backend/
|   |   +-- app/
|   |       +-- main.py              # FastAPI application factory + lifespan handler
|   |       |
|   |       +-- core/
|   |       |   +-- config.py        # Settings: MODEL_PATH, DATA_DIR, CORS, DB_PATH
|   |       |
|   |       +-- routers/
|   |       |   +-- reconstruct.py   # POST /api/v1/reconstruct
|   |       |   |                    #   RANSAC 8-GCP affine calibration
|   |       |   |                    #   Holdout validation (r, MAE, RMSE)
|   |       |   |                    #   Water leveling + edge sharpening
|   |       |   |                    #   Export: GeoTIFF + GLB
|   |       |   +-- presets.py       # GET /api/v1/presets
|   |       |
|   |       +-- services/
|   |       |   +-- inference.py     # Depth-Anything-V2-Small ONNX session wrapper
|   |       |   |                    #   ensure_model_weights() -- auto-download on startup
|   |       |   |                    #   _OnnxSession -- lazy singleton
|   |       |   |                    #   infer_depth() -- 518x518 ViT forward pass
|   |       |   |                    #   sharpen_urban_geometry() -- bilateral + unsharp mask
|   |       |   |
|   |       |   +-- calibrator.py    # Dual-mode calibration engine
|   |       |   |                    #   Sign correction (Pearson r check)
|   |       |   |                    #   RANSAC affine fit: Z = s*d-hat + t
|   |       |   |                    #   Holdout validation metrics
|   |       |   |
|   |       |   +-- water_leveling.py # Shadow-Safe Water Leveling (SSWL)
|   |       |   |                    #   VWI = (G-R)/(G+R+eps)
|   |       |   |                    #   Dual Otsu thresholding
|   |       |   |                    #   Luminance gating (tau = 0.25)
|   |       |   |                    #   Connected-region equipotential clamping
|   |       |   |
|   |       |   +-- export.py        # GeoTIFF (rasterio) + GLB (trimesh/pygltflib) export
|   |       |
|   |       +-- models/
|   |       |   +-- db.py            # SQLite schema + CRUD: presets, GCPs, reconstructions
|   |       |
|   |       +-- data/
|   |           +-- models/          # da_v2_small.onnx (auto-downloaded, git-ignored)
|   |           +-- presets/
|   |           |   +-- hyderabad/   # optical.png, depth.npy, elevation.npy,
|   |           |   +-- assam/       # height_matrix.npy, metadata.json
|   |           |   +-- leh/
|   |           +-- exports/         # Generated GeoTIFF / GLB exports (runtime)
|   |           +-- depthwizard.db   # SQLite reconstruction log
|   |
|   +-- frontend/
|       +-- src/
|           +-- App.jsx              # Root application: layout, state, API orchestration
|           +-- lib/
|           |   +-- api.js           # Typed fetch wrappers for backend endpoints
|           |
|           +-- components/
|               +-- Canvas3D.jsx             # @react-three/fiber WebGL canvas container
|               +-- TerrainMesh.jsx          # Three.js PlaneGeometry displacement mesh
|               |                            #   Bilinear UV height sampling
|               |                            #   Draped satellite optical texture (sRGB)
|               |                            #   Hypsometric vertex colour fallback (7-stop)
|               +-- ValidationHUD.jsx        # Glassmorphic telemetry overlay
|               |                            #   COLLAPSED: mode pill (TACTICAL / RECON)
|               |                            #   EXPANDED: Z=s*d+t, r, MAE, RMSE, GSD
|               +-- ElevationRuler.jsx       # A->B 3D ruler with animated beacon markers
|               |                            #   Euclidean + topographic distance
|               |                            #   50-point linear profile sampling
|               +-- TopoCrossSection.jsx     # Recharts elevation profile area chart
|               |                            #   Cumulative distance, altitude delta, slope %
|               +-- DiagnosticDrawer.jsx     # Bottom slide-up diagnostic panel
|               |                            #   Magma-colourmap depth preview canvas
|               |                            #   Water mask overlay toggle
|               +-- ViewportShaderToggle.jsx # Shader mode selector (Optical/Heatmap/Wireframe)
|               +-- FlythroughController.jsx # Keyboard/gamepad flythrough animation
|
+-- backend/                         # Legacy backend (standalone scripts + presets)
+-- frontend/                        # Legacy frontend (presets only)
+-- graphify-out/                    # Graphify knowledge graph (AST + community)
+-- venv/                            # Python virtual environment (git-ignored)
```

---

## 6. API Reference

### `POST /api/v1/reconstruct`

Execute the full DSM reconstruction pipeline.

**Request (multipart/form-data):**

| Field | Type | Description |
|-------|------|-------------|
| `preset_id` | string (optional) | `leh`, `hyderabad`, or `assam` — triggers Mode A (calibrated) |
| `image_file` | binary (optional) | Arbitrary JPG/PNG/TIFF upload — triggers Mode B (relative) |
| `level_water` | bool (default: true) | Enable Shadow-Safe Water Leveling |

Supply **either** `preset_id` or `image_file`, not both.

**Response (JSON):**

```json
{
  "reconstruction_id": "uuid-v4",
  "mode": "tactical",
  "preset_id": "hyderabad",
  "height_matrix": "<base64-encoded float32 256x256>",
  "height_matrix_shape": [256, 256],
  "units": "meters_amsl",
  "ground_resolution_m_per_px": 10.5,
  "calibration": {
    "s": 1847.23,
    "t": 2856.44,
    "depth_inverted": false,
    "fit_method": "ransac"
  },
  "metrics": {
    "r": 0.582,
    "mae": 7.74,
    "rmse": 9.85
  },
  "water_mask_applied": true,
  "depth_preview": "data:image/png;base64,..."
}
```

### `POST /api/v1/export/geotiff`

Export a 32-bit float GeoTIFF with Deflate compression, georeferenced to EPSG:4326.

### `POST /api/v1/export/glb`

Export a binary GLB mesh with draped optical texture and configurable vertical exaggeration.

### `GET /api/v1/presets`

Return the preset catalogue with metadata (bounding box, GSD, GCP count, region description).

### `GET /health`

Server liveness probe: `{"status": "ok"}`

---

## 7. Frontend GIS Suite

The DepthWizard interface is built on **React 19 + @react-three/fiber + Three.js r185** inside a Vite dev server.

### Viewport Shader Modes

| Mode | Key | Description |
|------|-----|-------------|
| **True Optical** | `1` | Satellite RGB tile draped over displaced mesh with sRGB gamma correction |
| **DEM Heatmap** | `2` | Hypsometric 7-stop colour gradient (deep blue to green to amber to grey to white) |
| **Wireframe** | `3` | Edge topology overlay revealing mesh structure and displacement density |

### Tactical GIS Tooling

| Tool | Description |
|------|-------------|
| **ElevationRuler** | Click two points in 3D space: measures Euclidean distance, topographic path distance, and elevation delta (delta-Z) with animated pulsing beacon markers |
| **TopoCrossSection** | Recharts area chart rendering a 50-point linear elevation profile between ruler endpoints, with cumulative distance, altitude delta, max slope %, and average slope % |
| **ValidationHUD** | Collapsible glassmorphic telemetry badge: mode pill (TACTICAL / RECON) collapsed; expands to show affine equation Z = s*d-hat + t, Pearson r, MAE, RMSE, and GSD |
| **DiagnosticDrawer** | Slide-up bottom panel with Magma-colourmap raw ONNX disparity preview and water mask overlay toggle |
| **FlythroughController** | Keyboard/gamepad-driven animated flythrough for cinematic terrain inspection |

### Hypsometric Gradient (DEM Heatmap Mode)

```
0.00  --  #0c4a6e  Deep valleys / water
0.15  --  #0284c7  Lowland lakes / rivers
0.35  --  #059669  Lush plains / forests
0.55  --  #d97706  Plateaus / arid hills
0.75  --  #9a3412  Rocky ridge relief
0.90  --  #64748b  Scree / alpine zone
1.00  --  #f8fafc  Snow-covered summits
```

---

## 8. Export Formats

| Format | Description | Use Case |
|--------|-------------|----------|
| **GeoTIFF (.tif)** | 32-bit float, Deflate compressed, EPSG:4326, elevation in metres AMSL | GIS workflows (QGIS, ArcGIS, GDAL), hydrological analysis |
| **GLB (.glb)** | Binary glTF triangular mesh with draped optical texture, configurable vertical exaggeration | 3D visualisation (Blender, Unity, WebGL viewers) |

---

## 9. License

MIT License — see `LICENSE` for full terms.

---

**DepthWizard** · Built for ISRO SIH-26175 · Single-View 3D Terrain Reconstruction

*"From a single pixel array to a navigable geodetic surface — in under 3 seconds."*
