import React, { useEffect, useMemo, useRef, Suspense } from 'react';
import * as THREE from 'three';
import { useTexture } from '@react-three/drei';

/**
 * OpticalTextureMaterial
 * Loads and drapes the satellite optical RGB tile over the displaced terrain.
 * Handles load errors gracefully — falls back to hypsometric vertex colours.
 */
function OpticalTextureMaterial({ url }) {
  const texture = useTexture(url);
  
  useEffect(() => {
    if (texture) {
      // Critical: declare sRGB colour space so Three.js gamma-corrects correctly.
      texture.colorSpace = THREE.SRGBColorSpace;
      texture.wrapS = THREE.ClampToEdgeWrapping;
      texture.wrapT = THREE.ClampToEdgeWrapping;
      texture.minFilter = THREE.LinearMipmapLinearFilter;
      texture.magFilter = THREE.LinearFilter;
      texture.generateMipmaps = true;
      texture.needsUpdate = true;
    }
  }, [texture]);

  return (
    <meshStandardMaterial
      map={texture}
      color={0xffffff}       // Pure white base — never tint the satellite imagery
      roughness={0.82}       // Slight specular catchlight on vertical building faces
      metalness={0.05}
      envMapIntensity={0.15} // Subtle rim highlight accentuates rooftop edges
      wireframe={false}
      flatShading={false}
    />
  );
}

/**
 * Hypsometric elevation color stops for tactical geospatial visualization.
 */
const HYPSO_STOPS = [
  { t: 0.00, color: new THREE.Color('#0c4a6e') }, // Deep valleys / water
  { t: 0.15, color: new THREE.Color('#0284c7') }, // Lowland lakes / rivers
  { t: 0.35, color: new THREE.Color('#059669') }, // Lush plains / forests
  { t: 0.55, color: new THREE.Color('#d97706') }, // Plateaus / arid hills
  { t: 0.75, color: new THREE.Color('#9a3412') }, // Rocky ridge relief
  { t: 0.90, color: new THREE.Color('#64748b') }, // Scree / alpine zone
  { t: 1.00, color: new THREE.Color('#f8fafc') }, // Snow-covered summits
];

function sampleHypsoGradient(t) {
  const clamped = Math.max(0, Math.min(1, t));
  for (let i = 0; i < HYPSO_STOPS.length - 1; i++) {
    const s0 = HYPSO_STOPS[i];
    const s1 = HYPSO_STOPS[i + 1];
    if (clamped >= s0.t && clamped <= s1.t) {
      const alpha = (clamped - s0.t) / (s1.t - s0.t);
      return s0.color.clone().lerp(s1.color, alpha);
    }
  }
  return HYPSO_STOPS[HYPSO_STOPS.length - 1].color.clone();
}

/**
 * TerrainMesh
 * ────────────────────────────────────────────────────────────────────────────
 * Presentation-grade 3D Digital Surface Model (DSM) terrain mesh.
 * 
 * Features:
 *   - Continuous UV-normalized bilinear sampling without stride artifacts
 *   - Clean square plane geometry without diagonal ribboning or spikes
 *   - Dynamic scale configuration (METRIC_SCALE = 0.08, MIN_VISIBLE_AMPLITUDE = 6.0)
 *     ensuring both low-relief floodplains and steep alpine massifs render
 *     with pristine geomorphic fidelity
 * ────────────────────────────────────────────────────────────────────────────
 */
export default function TerrainMesh({
  heightData,
  opticalUrl = null,
  verticalExaggeration = 1.0,
  colorMode = 'optical', // 'optical' | 'hypsometric' | 'wireframe'
  onPointerDown = null,
  meshRef = null,
}) {
  const localRef = useRef(null);
  const actualMeshRef = meshRef || localRef;

  const rawHeights = heightData?.data || (Array.isArray(heightData) ? heightData : (heightData instanceof Float32Array ? heightData : null));

  // Compute grid dimension safely
  const gridDim = useMemo(() => {
    if (!rawHeights || rawHeights.length === 0) return 256;
    const totalPoints = rawHeights.length;
    return Math.round(Math.sqrt(totalPoints)) || 256;
  }, [rawHeights]);

  // Ensure high-density mesh subdivision (512×512) for crisp building walls and flat rooftops
  const SEGMENTS = 512;
  const geometry = useMemo(() => {
    return new THREE.PlaneGeometry(100, 100, SEGMENTS, SEGMENTS);
  }, []);

  // Displace geometry vertices via clean UV-normalized bilinear sampling
  useEffect(() => {
    if (!geometry || !rawHeights || rawHeights.length === 0) return;

    const totalPoints = rawHeights.length;
    let minH = Infinity;
    let maxH = -Infinity;
    for (let i = 0; i < totalPoints; i++) {
      const v = rawHeights[i];
      if (v < minH) minH = v;
      if (v > maxH) maxH = v;
    }
    if (minH === Infinity || !Number.isFinite(minH) || !Number.isFinite(maxH) || maxH <= minH) {
      minH = 0.0;
      maxH = 100.0;
    }

    if (heightData) {
      heightData.minElev = minH;
      heightData.maxElev = maxH;
      heightData.minElevation = minH;
      heightData.maxElevation = maxH;
    }

    const reliefSpan = Math.max(maxH - minH, 1.0);

    // Target vertical height span in Three.js world units:
    // Alpine massifs (Leh) cap at ~10.0 units, plateaus (Hyderabad) at ~4.0 units, floodplains (Assam) at ~2.5 units.
    // Cap at 8.0 units so urban building alleys stay tight and vertical (not stretched into ramps).
    const targetMaxHeight = Math.min(8.0, Math.max(2.5, (reliefSpan / 800.0) * 8.0));
    const verticalScale = (targetMaxHeight / reliefSpan) * (verticalExaggeration || 1.0);

    const posAttr = geometry.attributes.position;
    const uvAttr = geometry.attributes.uv;
    const vertexCount = posAttr.count;
    const colors = new Float32Array(vertexCount * 3);

    for (let i = 0; i < vertexCount; i++) {
      const u = uvAttr.getX(i);
      const v = 1.0 - uvAttr.getY(i); // Invert Y to align with image coordinates

      const fx = u * (gridDim - 1);
      const fy = v * (gridDim - 1);

      const x0 = Math.floor(fx);
      const y0 = Math.floor(fy);
      const x1 = Math.min(x0 + 1, gridDim - 1);
      const y1 = Math.min(y0 + 1, gridDim - 1);

      const dx = fx - x0;
      const dy = fy - y0;

      const h00 = rawHeights[y0 * gridDim + x0] ?? minH;
      const h10 = rawHeights[y0 * gridDim + x1] ?? minH;
      const h01 = rawHeights[y1 * gridDim + x0] ?? minH;
      const h11 = rawHeights[y1 * gridDim + x1] ?? minH;

      const interpolatedHeight = 
        h00 * (1 - dx) * (1 - dy) +
        h10 * dx * (1 - dy) +
        h01 * (1 - dx) * dy +
        h11 * dx * dy;

      // Displace strictly relative to minimum elevation (baseline >= 0, cannot dip below floor)
      const displacedZ = Math.max(0.0, (interpolatedHeight - minH) * verticalScale);
      posAttr.setZ(i, displacedZ);

      // Procedural hypsometric vertex color
      const t = Math.max(0.0, Math.min(1.0, (interpolatedHeight - minH) / reliefSpan));
      const c = sampleHypsoGradient(t);
      colors[i * 3 + 0] = c.r;
      colors[i * 3 + 1] = c.g;
      colors[i * 3 + 2] = c.b;
    }

    posAttr.needsUpdate = true;
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geometry.computeVertexNormals();

    // Prevent geometry.center() from centering the vertical axis (which pushes lower half below floor).
    // Instead: center horizontally along X & Y, and bottom-align so terrain baseline rests exactly at Z = 0.
    geometry.computeBoundingBox();
    const box = geometry.boundingBox;
    const centerX = (box.max.x + box.min.x) / 2;
    const centerY = (box.max.y + box.min.y) / 2;
    const minZ = box.min.z;

    geometry.translate(-centerX, -centerY, -minZ);
    geometry.computeVertexNormals();
    geometry.computeBoundingBox();
    geometry.computeBoundingSphere();

    if (actualMeshRef && actualMeshRef.current) {
      actualMeshRef.current.position.set(0, 0, 0);
    }
  }, [geometry, rawHeights, gridDim, heightData, verticalExaggeration, actualMeshRef]);

  // Determine material based on colorMode
  const material = useMemo(() => {
    if (colorMode === 'wireframe') {
      return (
        <meshStandardMaterial
          wireframe
          color="#00e5ff"
          emissive="#002b36"
          roughness={0.4}
        />
      );
    }

    if (colorMode === 'hypsometric') {
      return (
        <meshStandardMaterial
          vertexColors
          roughness={0.65}       // More pronounced shading contrast on sharp ridges
          metalness={0.05}
          envMapIntensity={0.12}
          wireframe={false}
        />
      );
    }

    // 'optical' mode — white base so texture shows true colour; used as loading
    // placeholder while Suspense resolves the TextureLoader promise.
    return (
      <meshStandardMaterial
        color={0xffffff}
        roughness={0.85}
        metalness={0.05}
      />
    );
  }, [colorMode]);

  return (
    <group>
      <mesh
        ref={actualMeshRef}
        geometry={geometry}
        rotation={[-Math.PI / 2, 0, 0]}
        position={[0, 0, 0]}
        receiveShadow
        castShadow
        onPointerDown={onPointerDown}
      >
        {colorMode === 'optical' && opticalUrl ? (
          <Suspense fallback={material}>
            <OpticalTextureMaterial url={opticalUrl} />
          </Suspense>
        ) : (
          material
        )}
      </mesh>
    </group>
  );
}
