"""
scripts/smoke_test.py
──────────────────────
End-to-end smoke test.  Uses requests against a real uvicorn subprocess so
it works with any starlette/fastapi version without TestClient compatibility
concerns.

Tests:
  T1  GET  /health
  T2  GET  /api/v1/presets
  T3  GET  /api/v1/presets/leh
  T4  GET  /api/v1/presets/nonexistent   → 404
  T5  POST /api/v1/reconstruct           preset_id=leh        (tactical)
  T6  POST /api/v1/reconstruct           preset_id=hyderabad  (tactical)
  T7  POST /api/v1/reconstruct           preset_id=assam  level_water=false
  T8  POST /api/v1/reconstruct           arbitrary PNG upload (relative)
  T9  POST /api/v1/reconstruct           no inputs → 422
"""

from __future__ import annotations

import base64
import io
import struct
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image as PilImage

try:
    import requests  # type: ignore
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "requests", "-q"])
    import requests  # type: ignore

# ---------------------------------------------------------------------------
# Server management
# ---------------------------------------------------------------------------

_BACKEND = Path(__file__).resolve().parent.parent / "backend"
BASE_URL = "http://127.0.0.1:8765"

def _start_server() -> subprocess.Popen:
    """Start uvicorn in a subprocess and wait until it is ready."""
    env_path = Path(sys.executable).parent.parent   # venv root
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "app.main:app",
            "--host", "127.0.0.1",
            "--port", "8765",
            "--log-level", "warning",
        ],
        cwd=str(_BACKEND),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Poll until /health responds or timeout
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=2)
            if r.status_code == 200:
                print(f"  Server ready at {BASE_URL}")
                return proc
        except Exception:
            pass
        time.sleep(0.4)

    proc.kill()
    raise RuntimeError("Server did not start within 30 s")


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------

PASS = "PASS"
FAIL = "FAIL"
_results: list[tuple[str, bool, str]] = []


def check(label: str, condition: bool, note: str = "") -> None:
    tag = PASS if condition else f"FAIL"
    print(f"  [{tag}]  {label}" + (f"  <- {note}" if note else ""))
    _results.append((label, condition, note))


def assert_height_matrix(payload: dict, label: str) -> None:
    """Decode and validate the base64 height matrix."""
    encoded = payload.get("height_matrix", "")
    raw = base64.b64decode(encoded)
    if len(raw) < 8:
        check(f"{label}: height_matrix has content", False, f"only {len(raw)} bytes")
        return
    h_shape, w_shape = struct.unpack_from("<II", raw, 0)
    floats = np.frombuffer(raw, dtype="<f4", offset=8)
    check(f"{label}: matrix shape matches data",
          floats.size == h_shape * w_shape,
          f"{h_shape}x{w_shape}={h_shape*w_shape} expected, got {floats.size}")
    check(f"{label}: height_matrix_shape field",
          payload.get("height_matrix_shape") == [h_shape, w_shape])
    check(f"{label}: values are finite", bool(np.all(np.isfinite(floats))))


def _make_dummy_png(size: int = 64) -> bytes:
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    buf = io.BytesIO()
    PilImage.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def run_tests() -> None:
    print("\n" + "=" * 62)
    print("  DepthWizard Backend — End-to-End Smoke Test")
    print("=" * 62 + "\n")

    # T1 — health
    print("T1  GET /health")
    r = requests.get(f"{BASE_URL}/health")
    check("status 200", r.status_code == 200)
    j = r.json()
    check("status=ok", j.get("status") == "ok")
    check("inference_mode present", "inference_mode" in j, j.get("inference_mode", "?"))

    # T2 — list presets
    print("\nT2  GET /api/v1/presets")
    r = requests.get(f"{BASE_URL}/api/v1/presets")
    check("status 200", r.status_code == 200)
    j = r.json()
    presets = j.get("presets", [])
    check(">=3 presets", len(presets) >= 3, f"got {len(presets)}")
    check("each has gcp_counts",   all("gcp_counts"    in p for p in presets))
    check("each has thumbnail_url", all("thumbnail_url" in p for p in presets))

    # T3 — single preset
    print("\nT3  GET /api/v1/presets/leh")
    r = requests.get(f"{BASE_URL}/api/v1/presets/leh")
    check("status 200", r.status_code == 200)
    j = r.json()
    check("id=leh", j.get("id") == "leh")
    check("has_dem=True", j.get("has_dem") is True)
    gc = j.get("gcp_counts", {})
    check("15 train GCPs",   gc.get("train")   == 15, str(gc))
    check("15 holdout GCPs", gc.get("holdout") == 15, str(gc))

    # T4 — missing preset
    print("\nT4  GET /api/v1/presets/nonexistent")
    r = requests.get(f"{BASE_URL}/api/v1/presets/nonexistent")
    check("status 404", r.status_code == 404)

    # T5 — leh tactical
    print("\nT5  POST /api/v1/reconstruct  preset_id=leh")
    r = requests.post(
        f"{BASE_URL}/api/v1/reconstruct",
        data={"preset_id": "leh", "level_water": "true"},
    )
    check("status 200", r.status_code == 200,
          r.text[:200] if r.status_code != 200 else "")
    if r.status_code == 200:
        j = r.json()
        check("mode=tactical",      j.get("mode") == "tactical")
        check("units=meters_amsl",  j.get("units") == "meters_amsl")
        check("calibration present", j.get("calibration") is not None)
        cal = j.get("calibration", {})
        check("s > 0",              cal.get("s", -1) > 0, f"s={cal.get('s')}")
        check("depth_inverted present", "depth_inverted" in cal)
        check("fit_method present",     "fit_method" in cal)
        check("metrics present",    j.get("metrics") is not None)
        m = j.get("metrics", {})
        r_val = m.get("r")
        check("metric r in [-1,1]", r_val is not None and abs(r_val) <= 1.0,
              f"r={r_val}")
        check("water_mask_applied is bool", isinstance(j.get("water_mask_applied"), bool))
        check("ground_resolution=10.0", j.get("ground_resolution_m_per_px") == 10.0)
        assert_height_matrix(j, "leh")

    # T6 — hyderabad tactical
    print("\nT6  POST /api/v1/reconstruct  preset_id=hyderabad")
    r = requests.post(
        f"{BASE_URL}/api/v1/reconstruct",
        data={"preset_id": "hyderabad"},
    )
    check("status 200", r.status_code == 200)
    if r.status_code == 200:
        j = r.json()
        check("mode=tactical", j.get("mode") == "tactical")
        assert_height_matrix(j, "hyderabad")

    # T7 — assam, water off
    print("\nT7  POST /api/v1/reconstruct  preset_id=assam  level_water=false")
    r = requests.post(
        f"{BASE_URL}/api/v1/reconstruct",
        data={"preset_id": "assam", "level_water": "false"},
    )
    check("status 200", r.status_code == 200)
    if r.status_code == 200:
        j = r.json()
        check("water_mask_applied=False", j.get("water_mask_applied") is False)

    # T8 — arbitrary upload → relative
    print("\nT8  POST /api/v1/reconstruct  (PNG upload)")
    png = _make_dummy_png(64)
    r = requests.post(
        f"{BASE_URL}/api/v1/reconstruct",
        files={"image_file": ("test.png", io.BytesIO(png), "image/png")},
        data={"level_water": "true"},
    )
    check("status 200", r.status_code == 200,
          r.text[:200] if r.status_code != 200 else "")
    if r.status_code == 200:
        j = r.json()
        check("mode=relative",       j.get("mode") == "relative")
        check("units=relative_0_100", j.get("units") == "relative_0_100")
        check("calibration=null",    j.get("calibration") is None)
        check("metrics=null",        j.get("metrics") is None)
        check("ground_res=null",     j.get("ground_resolution_m_per_px") is None)
        assert_height_matrix(j, "upload")

    # T9 — no inputs
    print("\nT9  POST /api/v1/reconstruct  (no inputs)")
    r = requests.post(f"{BASE_URL}/api/v1/reconstruct")
    check("status 422", r.status_code == 422)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    total  = len(_results)
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = total - passed
    print(f"  Results: {passed}/{total} passed" + (f"  |  {failed} FAILED" if failed else "  ALL GREEN"))
    print("=" * 62 + "\n")

    if failed:
        print("FAILED checks:")
        for name, ok, note in _results:
            if not ok:
                print(f"  [FAIL]  {name}" + (f"  <- {note}" if note else ""))
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    server_proc = None
    try:
        print("Starting uvicorn server …")
        server_proc = _start_server()
        run_tests()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    finally:
        if server_proc is not None:
            server_proc.kill()
            server_proc.wait()
            print("Server stopped.")
