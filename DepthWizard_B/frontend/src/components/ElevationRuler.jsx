import React, { useState, useEffect, useRef } from 'react';
import * as THREE from 'three';
import { Html, Line } from '@react-three/drei';
import { useFrame } from '@react-three/fiber';

/**
 * Tactical Beacon Marker (Pin A / Pin B)
 */
function BeaconMarker({ position, label, color = '#00e5ff', elev = 0, units = 'm' }) {
  const beaconRef = useRef();

  useFrame(({ clock }) => {
    if (beaconRef.current) {
      const t = clock.getElapsedTime();
      beaconRef.current.rotation.y = t * 2.0;
      beaconRef.current.position.y = 1.0 + Math.sin(t * 3.0) * 0.2;
    }
  });

  return (
    <group position={[position.x, position.y, position.z]}>
      {/* Vertical laser beacon cylinder */}
      <mesh position={[0, 2.5, 0]}>
        <cylinderGeometry args={[0.08, 0.08, 5, 16]} />
        <meshBasicMaterial color={color} transparent opacity={0.6} />
      </mesh>

      {/* Pulsing floating beacon orb */}
      <mesh ref={beaconRef} position={[0, 1.0, 0]}>
        <sphereGeometry args={[0.4, 16, 16]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={0.8}
          roughness={0.2}
        />
      </mesh>

      {/* Ground radar ring */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.1, 0]}>
        <ringGeometry args={[0.6, 0.8, 24]} />
        <meshBasicMaterial color={color} transparent opacity={0.7} side={THREE.DoubleSide} />
      </mesh>

      {/* Tactical HUD Pin Label */}
      <Html position={[0, 3.8, 0]} center distanceFactor={70}>
        <div className="pointer-events-none select-none px-2 py-1 rounded bg-slate-950/90 border border-slate-700 text-white font-mono text-[10px] shadow-hud-glow whitespace-nowrap flex items-center gap-1.5 backdrop-blur-md">
          <span className="w-1.5 h-1.5 rounded-full animate-ping" style={{ backgroundColor: color }} />
          <strong style={{ color }}>{label}</strong>
          <span className="text-slate-400">|</span>
          <span>{Math.round(elev * 10) / 10} {units}</span>
        </div>
      </Html>
    </group>
  );
}

/**
 * ElevationRuler
 * ────────────────────────────────────────────────────────────────────────────
 * Interactive raycasting tool for point-to-point elevation profile sampling.
 * 
 * Extracts UVs on terrain click:
 *   col = Math.min(Math.floor(uv.x * (width - 1)), width - 1);
 *   row = Math.min(Math.floor((1.0 - uv.y) * (height - 1)), height - 1);
 * 
 * Computes 50-point linear profile & slope metrics between Pin A and Pin B.
 * ────────────────────────────────────────────────────────────────────────────
 */
export default function ElevationRuler({
  heightData,
  groundResolution = null,
  units = 'meters_amsl',
  mode = 'tactical',
  active = true,
  onProfileGenerated = null,
  pinA = null,
  pinB = null,
  setPinA = null,
  setPinB = null,
}) {
  const isTactical = mode === 'tactical';
  const unitLabel = isTactical ? 'm' : 'rel';

  // Compute and emit 50-point profile slice when both pins are set
  useEffect(() => {
    if (!pinA || !pinB || !heightData || !heightData.data) {
      if (onProfileGenerated) onProfileGenerated(null);
      return;
    }

    const { width, height, data } = heightData;
    const N = 50;
    const points = [];

    const colDiff = pinB.col - pinA.col;
    const rowDiff = pinB.row - pinA.row;

    // Ground resolution per pixel in meters (defaults to 1.0 for relative mode)
    const gsd = isTactical && typeof groundResolution === 'number' && groundResolution > 0
      ? groundResolution
      : 1.0;

    const pixelDist = Math.hypot(colDiff, rowDiff);
    const totalDistanceMeters = pixelDist * gsd;
    const elevDeltaMeters = pinB.elev - pinA.elev;

    let maxSlope = 0;
    let sumSlope = 0;
    let minElev = Infinity;
    let maxElev = -Infinity;

    for (let i = 0; i < N; i++) {
      const t = i / (N - 1);
      const sampleCol = Math.min(Math.max(Math.round(pinA.col + t * colDiff), 0), width - 1);
      const sampleRow = Math.min(Math.max(Math.round(pinA.row + t * rowDiff), 0), height - 1);
      const elev = data[sampleRow * width + sampleCol];

      if (elev < minElev) minElev = elev;
      if (elev > maxElev) maxElev = elev;

      const dist = t * totalDistanceMeters;

      let slope = 0;
      if (i > 0) {
        const prevDist = ((i - 1) / (N - 1)) * totalDistanceMeters;
        const prevCol = Math.min(Math.max(Math.round(pinA.col + ((i - 1) / (N - 1)) * colDiff), 0), width - 1);
        const prevRow = Math.min(Math.max(Math.round(pinA.row + ((i - 1) / (N - 1)) * rowDiff), 0), height - 1);
        const prevElev = data[prevRow * width + prevCol];
        const segmentDist = dist - prevDist;

        if (segmentDist > 0) {
          slope = ((elev - prevElev) / segmentDist) * 100;
        }
      }

      if (Math.abs(slope) > maxSlope) maxSlope = Math.abs(slope);
      sumSlope += Math.abs(slope);

      points.push({
        index: i,
        distance: dist,
        elevation: elev,
        slope: Math.round(slope * 10) / 10,
        col: sampleCol,
        row: sampleRow,
      });
    }

    const avgSlope = N > 1 ? sumSlope / (N - 1) : 0;
    const overallSlopePercent = totalDistanceMeters > 0
      ? (elevDeltaMeters / totalDistanceMeters) * 100
      : 0;

    if (onProfileGenerated) {
      onProfileGenerated({
        points,
        totalDistance: totalDistanceMeters,
        elevationDelta: elevDeltaMeters,
        maxSlope: Math.round(maxSlope * 10) / 10,
        avgSlope: Math.round(avgSlope * 10) / 10,
        overallSlope: Math.round(overallSlopePercent * 10) / 10,
        minElevation: minElev,
        maxElevation: maxElev,
        startElev: pinA.elev,
        endElev: pinB.elev,
        startPoint: { col: pinA.col, row: pinA.row },
        endPoint: { col: pinB.col, row: pinB.row },
      });
    }
  }, [pinA, pinB, heightData, groundResolution, isTactical, onProfileGenerated]);

  // Points for 3D connecting line (lifted slightly by +0.3Y to prevent z-fighting)
  const linePoints = pinA && pinB ? [
    [pinA.worldPoint.x, pinA.worldPoint.y + 0.3, pinA.worldPoint.z],
    [pinB.worldPoint.x, pinB.worldPoint.y + 0.3, pinB.worldPoint.z],
  ] : null;

  return (
    <group>
      {/* Pin A Marker (Cyan) */}
      {pinA && (
        <BeaconMarker
          position={pinA.worldPoint}
          label="PIN A"
          color="#00e5ff"
          elev={pinA.elev}
          units={unitLabel}
        />
      )}

      {/* Pin B Marker (Amber) */}
      {pinB && (
        <BeaconMarker
          position={pinB.worldPoint}
          label="PIN B"
          color="#f59e0b"
          elev={pinB.elev}
          units={unitLabel}
        />
      )}

      {/* 3D Connecting Trajectory Line */}
      {linePoints && (
        <Line
          points={linePoints}
          color="#38bdf8"
          lineWidth={2.5}
          dashed
          dashScale={2}
          dashSize={1}
          gapSize={0.5}
        />
      )}
    </group>
  );
}

/**
 * Utility helper to extract matrix coordinates and elevation from Three.js raycast intersection.
 * Applies the mandated V-axis inversion:
 *   col = Math.min(Math.floor(uv.x * (width - 1)), width - 1);
 *   row = Math.min(Math.floor((1.0 - uv.y) * (height - 1)), height - 1);
 */
export function extractElevationIntersection(intersection, heightData) {
  if (!intersection || !intersection.uv || !heightData || !heightData.data) {
    return null;
  }

  const { uv, point } = intersection;
  const { width, height, data } = heightData;

  const col = Math.min(Math.floor(uv.x * (width - 1)), width - 1);
  const row = Math.min(Math.floor((1.0 - uv.y) * (height - 1)), height - 1);
  const elev = data[row * width + col] ?? 0;

  return {
    worldPoint: { x: point.x, y: point.y, z: point.z },
    uv: { x: uv.x, y: uv.y },
    col,
    row,
    elev,
  };
}
