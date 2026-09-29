/**
 * frontend/src/lib/api.js
 * ────────────────────────────────────────────────────────────────────────────
 * ISRO DepthWizard / TerraMono API Client & Binary Matrix Decoder
 * 
 * Provides:
 *   - fetchPresets(): Lists available pre-calibrated regional presets
 *   - fetchPreset(presetId): Fetches individual preset metadata and GCP counts
 *   - reconstructTerrain(params): Dispatches single-view 3D reconstruction
 *   - decodeHeightMatrix(base64Str): Unpacks raw <II 8-byte header & <f4 float array
 * ────────────────────────────────────────────────────────────────────────────
 */

const API_BASE = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/+$/, '');

/**
 * Fetch list of regional presets (Leh, Hyderabad, Assam, etc.)
 * @returns {Promise<Array<Object>>} List of enriched preset objects
 */
export async function fetchPresets() {
  const response = await fetch(`${API_BASE}/api/v1/presets`, {
    method: 'GET',
    headers: {
      'Accept': 'application/json',
    },
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`Failed to fetch presets (${response.status}): ${errText}`);
  }

  const json = await response.json();
  return json.presets || [];
}

/**
 * Fetch detailed metadata for a single preset
 * @param {string} presetId - Preset region ID
 * @returns {Promise<Object>} Preset detail object with GCP counts
 */
export async function fetchPreset(presetId) {
  const response = await fetch(`${API_BASE}/api/v1/presets/${encodeURIComponent(presetId)}`, {
    method: 'GET',
    headers: {
      'Accept': 'application/json',
    },
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`Failed to fetch preset '${presetId}' (${response.status}): ${errText}`);
  }

  return await response.json();
}

/**
 * Dispatches 3D terrain reconstruction request to backend.
 * 
 * @param {Object} params
 * @param {string} [params.preset_id] - Regional preset ID (e.g. 'leh', 'hyderabad', 'assam')
 * @param {string} [params.presetId] - CamelCase alias for preset_id
 * @param {File|Blob} [params.image_file] - Arbitrary optical satellite/aerial image
 * @param {File|Blob} [params.imageFile] - CamelCase alias for image_file
 * @param {boolean} [params.level_water=true] - Apply shadow-safe water leveling
 * @param {boolean} [params.levelWater=true] - CamelCase alias for level_water
 * 
 * @returns {Promise<Object>} Reconstruction payload matching PRD Section 10 contract:
 * {
 *   reconstruction_id: string,
 *   mode: 'tactical' | 'relative',
 *   preset_id: string | null,
 *   height_matrix: string (base64 raw binary),
 *   height_matrix_shape: [number, number],
 *   units: 'meters_amsl' | 'relative_0_100',
 *   ground_resolution_m_per_px: number | null,
 *   calibration: { s: number, t: number, depth_inverted: boolean, fit_method: string } | null,
 *   metrics: { r: number, mae: number, rmse: number } | null,
 *   water_mask_applied: boolean
 * }
 */
export async function reconstructTerrain(params = {}) {
  const presetId = params.preset_id || params.presetId;
  const imageFile = params.image_file || params.imageFile;
  const levelWater = params.level_water !== undefined
    ? params.level_water
    : (params.levelWater !== undefined ? params.levelWater : true);

  if (!presetId && !imageFile) {
    throw new Error("reconstructTerrain requires either 'preset_id' or 'image_file'.");
  }

  const formData = new FormData();

  if (presetId) {
    formData.append('preset_id', presetId);
  } else if (imageFile) {
    formData.append('image_file', imageFile);
  }

  formData.append('level_water', String(Boolean(levelWater)));

  const response = await fetch(`${API_BASE}/api/v1/reconstruct`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    let errorDetail;
    try {
      const errJson = await response.json();
      errorDetail = errJson.detail || JSON.stringify(errJson);
    } catch {
      errorDetail = await response.text();
    }
    throw new Error(`Reconstruction failed [${response.status}]: ${errorDetail}`);
  }

  return await response.json();
}

/**
 * Binary base64 height-matrix decoder.
 * Unpacks the raw 8-byte header (<II: two uint32 LE values for height, width)
 * and reads the contiguous <f4 (float32 little-endian) array into a Float32Array.
 * 
 * @param {string} base64Str - Base64 encoded binary payload from backend
 * @returns {{
 *   height: number,
 *   width: number,
 *   data: Float32Array,
 *   minElevation: number,
 *   maxElevation: number
 * }}
 */
export function decodeHeightMatrix(base64Str) {
  if (!base64Str || typeof base64Str !== 'string') {
    throw new Error('decodeHeightMatrix: valid base64 string required.');
  }

  const cleanB64 = base64Str.trim();
  const binaryString = atob(cleanB64);
  const totalBytes = binaryString.length;

  if (totalBytes < 8) {
    throw new Error(`decodeHeightMatrix: buffer too short (${totalBytes} bytes < 8 bytes header).`);
  }

  // Copy raw bytes into Uint8Array
  const uint8Buffer = new Uint8Array(totalBytes);
  for (let i = 0; i < totalBytes; i++) {
    uint8Buffer[i] = binaryString.charCodeAt(i);
  }

  // Read 8-byte header: uint32 LE height, uint32 LE width
  const dataView = new DataView(uint8Buffer.buffer, uint8Buffer.byteOffset, uint8Buffer.byteLength);
  const height = dataView.getUint32(0, true);
  const width = dataView.getUint32(4, true);

  const expectedFloats = height * width;
  const expectedDataBytes = expectedFloats * 4;

  if (totalBytes - 8 < expectedDataBytes) {
    console.warn(
      `decodeHeightMatrix: expected ${expectedDataBytes} bytes for ${height}x${width} matrix, but got ${totalBytes - 8} bytes.`
    );
  }

  // Slice buffer to ensure 4-byte byteOffset alignment for Float32Array
  const floatByteOffset = uint8Buffer.byteOffset + 8;
  const alignedBuffer = uint8Buffer.buffer.slice(floatByteOffset, floatByteOffset + expectedDataBytes);
  const floatArray = new Float32Array(alignedBuffer);

  // Compute terrain elevation bounds
  let minElev = Infinity;
  let maxElev = -Infinity;
  const len = floatArray.length;
  for (let i = 0; i < len; i++) {
    const val = floatArray[i];
    if (val < minElev) minElev = val;
    if (val > maxElev) maxElev = val;
  }

  return {
    height,
    width,
    data: floatArray,
    minElevation: minElev === Infinity ? 0 : minElev,
    maxElevation: maxElev === -Infinity ? 100 : maxElev,
    minElev: minElev === Infinity ? 0 : minElev,
    maxElev: maxElev === -Infinity ? 100 : maxElev,
  };
}

/**
 * Returns the optical texture URL for a preset region.
 *
 * Priority: Vite public directory (/presets/<id>/optical.png) which avoids
 * cross-origin issues during `npm run dev`. The backend also serves these
 * at /static/presets/<id>/optical.png as a fallback.
 *
 * @param {string} presetId - Preset region ID (e.g. 'leh', 'hyderabad', 'assam')
 * @returns {string} URL safe for THREE.TextureLoader
 */
export function getPresetImageUrl(presetId) {
  // Resolve from Vite public/presets/<id>/optical.png — no CORS, no backend dep.
  return `/presets/${encodeURIComponent(presetId)}/optical.png`;
}

/**
 * Downloads a 32-bit float GeoTIFF file from the backend export service.
 */
export async function exportGeoTIFF({ preset_id = null, height_matrix }) {
  if (!height_matrix) throw new Error('height_matrix is required for GeoTIFF export.');

  const form = new FormData();
  if (preset_id) form.append('preset_id', preset_id);
  form.append('height_matrix', height_matrix);

  const response = await fetch(`${API_BASE}/api/v1/export/geotiff`, {
    method: 'POST',
    body: form,
  });

  if (!response.ok) {
    const err = await response.text();
    throw new Error(`GeoTIFF export failed (${response.status}): ${err}`);
  }

  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `depthwizard_${preset_id || 'dsm'}.tif`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/**
 * Downloads a binary 3D GLB file from the backend export service.
 */
export async function exportGLB({ preset_id = null, height_matrix, vertical_exaggeration = 1.0 }) {
  if (!height_matrix) throw new Error('height_matrix is required for GLB export.');

  const form = new FormData();
  if (preset_id) form.append('preset_id', preset_id);
  form.append('height_matrix', height_matrix);
  form.append('vertical_exaggeration', String(vertical_exaggeration));

  const response = await fetch(`${API_BASE}/api/v1/export/glb`, {
    method: 'POST',
    body: form,
  });

  if (!response.ok) {
    const err = await response.text();
    throw new Error(`GLB export failed (${response.status}): ${err}`);
  }

  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `depthwizard_${preset_id || 'terrain'}.glb`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

export { API_BASE };

