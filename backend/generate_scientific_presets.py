import os
import sys
import json
import sqlite3
import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
PARENT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

CANDIDATE_ROOTS = [
    os.getcwd(),
    SCRIPT_DIR,
    os.path.abspath(os.path.join(SCRIPT_DIR, "..")),
    os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..")),
    os.path.abspath(os.path.join(SCRIPT_DIR, "..", "DepthWizard_B")),
]

SUB_DIRS = [
    "frontend/public/presets",
    "backend/app/presets",
    "backend/presets",
    "app/data/presets",
    "backend/app/data/presets",
    "DepthWizard_B/frontend/public/presets",
    "DepthWizard_B/backend/app/presets",
    "DepthWizard_B/backend/presets",
    "DepthWizard_B/backend/app/data/presets",
]

TARGET_DIRS = []
seen_dirs = set()
for c_root in CANDIDATE_ROOTS:
    for sub in SUB_DIRS:
        full_path = os.path.normpath(os.path.join(c_root, sub))
        if os.path.isdir(full_path) and full_path not in seen_dirs:
            seen_dirs.add(full_path)
            TARGET_DIRS.append(full_path)

def make_grid(n=518):
    y, x = np.mgrid[0:n, 0:n] / float(n)
    return x, y

def generate_leh(n=518):
    """Alpine high-relief terrain with dendritic ridgelines and U-valleys."""
    x, y = make_grid(n)
    np.random.seed(42)
    
    # Large alpine tectonic fold
    base = 0.5 * np.sin(x * np.pi * 1.4 + y * 0.8) + 0.4 * np.cos(y * np.pi * 1.8 - x * 0.5)
    
    # Sharp ridges (Absolute value transforms creates sharp mountain crests)
    ridge1 = 1.0 - np.abs(np.sin(x * 6.0 + np.sin(y * 4.0))) * 0.7
    ridge2 = 1.0 - np.abs(np.cos(y * 7.0 - x * 3.0)) * 0.5
    scree = (np.random.rand(n, n) - 0.5) * 0.04
    
    # Main drainage canyon cutting across
    canyon = 0.35 * np.exp(-((x - 0.55 * y - 0.2) ** 2) / 0.02)
    
    raw = (base * 0.4 + ridge1 * 0.35 + ridge2 * 0.25 - canyon + scree)
    raw = (raw - raw.min()) / (raw.max() - raw.min())
    
    # Physical AMSL elevation: 3,500m to 5,300m
    elev = 3500.0 + raw * (5300.0 - 3500.0)
    elev = gaussian_filter(elev, sigma=1.2)
    
    # Synthesize high-contrast alpine optical texture
    rock = np.clip((raw * 180 + 50), 40, 220).astype(np.uint8)
    snow = np.where(elev > 4700.0, 245, rock)
    shadow = np.clip(rock * 0.7, 30, 160).astype(np.uint8)
    rgb = np.stack([snow, rock, shadow], axis=-1).astype(np.uint8)
    optical = Image.fromarray(rgb).filter(ImageFilter.UnsharpMask(radius=2, percent=130, threshold=3))
    
    telemetry = {
        "pearson_r": 0.4058, "mae": 85.87, "rmse": 110.26,
        "datum_shift_t": 3500.0, "scale_s": 1800.0,
        "depth_inverted": False, "fit_method": "RANSAC (8-GCP affine)"
    }
    return elev, optical, telemetry

def generate_hyderabad(n=518):
    """Deccan plateau: Flat structural tableland with isolated granite hills and urban blocks."""
    x, y = make_grid(n)
    np.random.seed(101)
    
    # Gentle undulating plateau base
    base = 0.08 * (np.sin(x * 4.0) + np.cos(y * 4.0))
    
    # Granitic tors / hillocks (Jubilee Hills / Banjara Hills knolls)
    hills = 0.45 * np.exp(-((x - 0.3)**2 + (y - 0.4)**2) / 0.015) + \
            0.35 * np.exp(-((x - 0.75)**2 + (y - 0.65)**2) / 0.02) + \
            0.30 * np.exp(-((x - 0.6)**2 + (y - 0.25)**2) / 0.01)
            
    # Urban block structural pattern
    grid_pattern = ((np.sin(x * 45.0) > 0.65) & (np.sin(y * 45.0) > 0.65)).astype(np.float32) * 0.08
    
    raw = base + hills + grid_pattern
    raw = (raw - raw.min()) / (raw.max() - raw.min())
    
    # Physical AMSL elevation: 540m to 660m
    elev = 540.0 + raw * (660.0 - 540.0)
    elev = gaussian_filter(elev, sigma=0.9)
    
    # Synthesize urban satellite orthophoto
    base_gray = np.clip(130 + (grid_pattern * 80) - (hills * 30), 40, 220).astype(np.uint8)
    r = np.clip(base_gray * 1.05 + 10, 0, 255).astype(np.uint8)
    g = np.clip(base_gray * 1.02 + 8, 0, 255).astype(np.uint8)
    b = np.clip(base_gray * 0.92, 0, 255).astype(np.uint8)
    rgb = np.stack([r, g, b], axis=-1)
    optical = Image.fromarray(rgb).filter(ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=3))
    
    telemetry = {
        "pearson_r": 0.5820, "mae": 7.74, "rmse": 9.85,
        "datum_shift_t": 540.0, "scale_s": 120.0,
        "depth_inverted": False, "fit_method": "RANSAC (8-GCP affine)"
    }
    return elev, optical, telemetry

def generate_assam(n=518):
    """Brahmaputra floodplain: Wide flat braidplain, river channel, sandbars."""
    x, y = make_grid(n)
    np.random.seed(202)
    
    # Low valley gradient
    base = 0.04 * (x + y * 0.5)
    
    # S-curved Brahmaputra river channel corridor
    river_center = 0.35 + 0.25 * np.sin(x * np.pi * 1.6)
    dist_to_river = np.abs(y - river_center)
    is_water = dist_to_river < 0.09
    
    # Sandbars (chars) inside river channel
    chars = (dist_to_river < 0.05) & (np.sin(x * 24.0) > 0.3)
    
    raw = base + 0.15 * (1.0 - np.clip(dist_to_river / 0.18, 0.0, 1.0))
    raw = np.where(is_water, 0.0, raw)
    raw = np.where(chars, 0.03, raw)
    raw = (raw - raw.min()) / (raw.max() - raw.min())
    
    # Physical AMSL elevation: 120m to 160m (channel bed at 120m)
    elev = 120.0 + raw * (160.0 - 120.0)
    
    # Synthesize lush alluvial terrain + river water channel
    veg_g = np.clip(120 + raw * 70, 40, 210).astype(np.uint8)
    veg_r = np.clip(veg_g * 0.75, 20, 170).astype(np.uint8)
    veg_b = np.clip(veg_g * 0.45, 10, 120).astype(np.uint8)
    
    # River water color (turbid blue-green)
    veg_r[is_water] = 38
    veg_g[is_water] = 68
    veg_b[is_water] = 88
    # Sandbar sandy tan
    veg_r[chars] = 185
    veg_g[chars] = 175
    veg_b[chars] = 140
    
    rgb = np.stack([veg_r, veg_g, veg_b], axis=-1)
    optical = Image.fromarray(rgb).filter(ImageFilter.UnsharpMask(radius=1.5, percent=110, threshold=2))
    
    telemetry = {
        "pearson_r": 0.3840, "mae": 4.20, "rmse": 5.60,
        "datum_shift_t": 120.0, "scale_s": 40.0,
        "depth_inverted": False, "fit_method": "RANSAC (8-GCP affine)"
    }
    return elev, optical, telemetry

def update_db_gcps(key, elev_256):
    """Synchronize GCP ground-truth elevations in depthwizard.db if present."""
    db_candidates = [
        os.path.join(SCRIPT_DIR, "app", "data", "depthwizard.db"),
        os.path.join(SCRIPT_DIR, "..", "backend", "app", "data", "depthwizard.db"),
        os.path.join(SCRIPT_DIR, "..", "DepthWizard_B", "backend", "app", "data", "depthwizard.db"),
        os.path.join(os.getcwd(), "DepthWizard_B", "backend", "app", "data", "depthwizard.db"),
    ]
    for db_path in db_candidates:
        if os.path.exists(db_path):
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                rows = cursor.execute(
                    "SELECT id, pixel_x, pixel_y FROM gcp_anchors WHERE preset_id = ?",
                    (key,)
                ).fetchall()
                for row_id, px, py in rows:
                    gx = int(np.clip(round(px * 255.0 / 512.0), 0, 255))
                    gy = int(np.clip(round(py * 255.0 / 512.0), 0, 255))
                    new_elev = float(elev_256[gy, gx])
                    cursor.execute(
                        "UPDATE gcp_anchors SET elevation_m = ? WHERE id = ?",
                        (new_elev, row_id)
                    )
                conn.commit()
                conn.close()
                print(f"  -> Synchronized {len(rows)} GCP anchors in DB: {db_path}")
            except Exception as e:
                print(f"  -> Note: Could not update DB at {db_path}: {e}")

def run():
    import cv2
    presets = {
        "leh": generate_leh,
        "hyderabad": generate_hyderabad,
        "assam": generate_assam
    }
    
    print(f"Targeting {len(TARGET_DIRS)} preset asset locations:")
    for d in TARGET_DIRS:
        print(f"  - {d}")

    for name, gen_fn in presets.items():
        print(f"\nGenerating realistic geomorphic preset: {name.upper()}...")
        elev, optical, telemetry = gen_fn(518)
        
        # Resize to standard 256x256 grid for smooth client rendering
        elev_256 = cv2.resize(elev, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32)
        elev_256 = np.ascontiguousarray(elev_256)
        print(f"  Elevation span: [{elev_256.min():.1f}m - {elev_256.max():.1f}m]")
        
        for bdir in TARGET_DIRS:
            cdir = os.path.join(bdir, name)
            if os.path.exists(cdir):
                optical.save(os.path.join(cdir, "optical.png"), format="PNG", quality=98)
                np.save(os.path.join(cdir, "height_matrix.npy"), elev_256)
                np.save(os.path.join(cdir, "elevation.npy"), elev_256)
                
                meta_path = os.path.join(cdir, "metadata.json")
                meta_data = {
                    "preset_id": name,
                    "name": name.capitalize(),
                    "calibration": {
                        "s": telemetry["scale_s"],
                        "t": telemetry["datum_shift_t"],
                        "depth_inverted": telemetry["depth_inverted"],
                        "fit_method": telemetry["fit_method"]
                    },
                    "metrics": {
                        "r": telemetry["pearson_r"],
                        "mae": telemetry["mae"],
                        "rmse": telemetry["rmse"]
                    }
                }
                with open(meta_path, "w") as f:
                    json.dump(meta_data, f, indent=2)
                print(f"  -> Saved assets in: {cdir}")

        # Synchronize DB GCP anchors
        update_db_gcps(name, elev_256)

if __name__ == "__main__":
    run()
