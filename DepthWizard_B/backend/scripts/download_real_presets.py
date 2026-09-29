import os
import sys
import glob
import requests
from PIL import Image
from io import BytesIO

# Coordinates centered on target regions (Bounding boxes: [min_lon, min_lat, max_lon, max_lat])
REGIONS = {
    "hyderabad": {
        # HITEC City / Durgam Cheruvu Urban Grid
        "bbox": "78.375,17.425,78.395,17.445",
        "name": "Hyderabad Urban Plateau"
    },
    "leh": {
        # Leh Valley and rugged Himalayan ridges
        "bbox": "77.560,34.140,77.600,34.180",
        "name": "Leh Ladakh Alpine Ridge"
    },
    "assam": {
        # Brahmaputra River Basin floodplain channels
        "bbox": "91.700,26.180,91.750,26.220",
        "name": "Assam Brahmaputra Valley"
    }
}

ESRI_EXPORT_URL = "https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/export"

# Dynamically locate project directories relative to script location and CWD
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(SCRIPT_DIR)
REPO_ROOT = os.path.dirname(BACKEND_DIR)

# Comprehensive target directory candidates
POTENTIAL_DESTS = [
    # Canonical frontend public presets (served by Vite)
    os.path.join(REPO_ROOT, "frontend", "public", "presets"),
    # Backend database tile path targets
    os.path.join(REPO_ROOT, "backend", "app", "data", "presets"),
    os.path.join(REPO_ROOT, "backend", "presets"),
    # Additional preset directories if present
    os.path.join(REPO_ROOT, "backend", "app", "assets", "presets"),
    os.path.join(REPO_ROOT, "backend", "assets", "presets"),
    # Relative paths based on CWD
    os.path.abspath("frontend/public/presets"),
    os.path.abspath("backend/app/assets/presets"),
    os.path.abspath("backend/assets/presets"),
    os.path.abspath("backend/app/data/presets"),
    os.path.abspath("backend/presets"),
]

# De-duplicate while preserving order
DEST_DIRS = []
for p in POTENTIAL_DESTS:
    canon = os.path.normpath(os.path.abspath(p))
    if canon not in DEST_DIRS:
        DEST_DIRS.append(canon)

def fetch_satellite_image(bbox, width=1024, height=1024):
    """
    Fetch genuine satellite imagery from the public ESRI World Imagery MapServer.
    Using 1024x1024 resolution yields crisp, georeferenced satellite photography
    with file sizes between 1.2 MB and 2.5 MB per scene.
    """
    params = {
        "bbox": bbox,
        "bboxSR": "4326",
        "layers": "",
        "layerDefs": "",
        "size": f"{width},{height}",
        "imageSR": "4326",
        "format": "png",
        "transparent": "false",
        "f": "image"
    }
    headers = {
        "User-Agent": "DepthWizard-Satellite-Fetcher/1.0"
    }
    response = requests.get(ESRI_EXPORT_URL, params=params, headers=headers, timeout=45)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")

def clean_stale_precomputed_arrays(region_dir: str):
    """
    Remove stale precomputed depth and height matrices so that the backend
    recalculates genuine depth using Depth Anything V2 from the new satellite
    imagery on subsequent runs.
    """
    stale_patterns = ["dsm.npy", "height_matrix.npy", "depth.npy", "elevation.npy"]
    for pattern in stale_patterns:
        target_path = os.path.join(region_dir, pattern)
        if os.path.exists(target_path):
            try:
                os.remove(target_path)
                print(f"    [Cleaned stale] Removed {pattern}")
            except Exception as e:
                print(f"    [Warning] Could not remove {pattern}: {e}")

def main():
    print("=" * 70)
    print("DepthWizard (SIH26175) — Real Satellite Imagery Ingestion")
    print("=" * 70)
    
    for region_id, info in REGIONS.items():
        print(f"\nFetching genuine satellite imagery for {info['name']} ({region_id})...")
        try:
            img = fetch_satellite_image(info["bbox"], width=1024, height=1024)
            print(f"  Downloaded optical tile: {img.size[0]}x{img.size[1]} px")
            
            for base_dir in DEST_DIRS:
                # Ensure target is within an existing branch or create parent
                parent = os.path.dirname(base_dir)
                if os.path.exists(parent):
                    region_dir = os.path.join(base_dir, region_id)
                    os.makedirs(region_dir, exist_ok=True)
                    out_path = os.path.join(region_dir, "optical.png")
                    img.save(out_path, format="PNG")
                    size_kb = os.path.getsize(out_path) // 1024
                    size_mb = os.path.getsize(out_path) / (1024 * 1024)
                    print(f"  -> Saved {out_path} ({size_kb} KB / {size_mb:.2f} MB)")
                    
                    # Remove stale precomputed matrices
                    clean_stale_precomputed_arrays(region_dir)
        except Exception as e:
            print(f"  [ERROR] Failed fetching {region_id}: {e}")

    print("\n" + "=" * 70)
    print("Asset Ingestion Complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
