import React, { useRef, useEffect, useMemo } from 'react';
import * as THREE from 'three';
import { useFrame, useThree } from '@react-three/fiber';
import { Line } from '@react-three/drei';

/**
 * Reconnaissance Flight Spline Corridor
 * Sweeps across the 100x100 terrain (entry from SW edge, swooping low through
 * the central valley, exiting high over NE ridge).
 */
const SPLINE_WAYPOINTS = [
  new THREE.Vector3(-44, 28, 44),   // Ingress: high-altitude approach
  new THREE.Vector3(-24, 15, 24),   // Descent: valley entrance
  new THREE.Vector3(-2, 7.5, 2),    // Recon pass: low valley skimming
  new THREE.Vector3(22, 14, -20),   // Climb: gorge ascent
  new THREE.Vector3(36, 22, -36),   // Ridge surveillance
  new THREE.Vector3(45, 30, -45),   // Egress: exit AOI
];

/**
 * FlythroughController
 * ────────────────────────────────────────────────────────────────────────────
 * Forward reconnaissance drone flight controller.
 * 
 * - Disables OrbitControls when active.
 * - Interpolates camera along Catmull-Rom spline with look-ahead targeting.
 * - Displays 3D flight path ribbon.
 * ────────────────────────────────────────────────────────────────────────────
 */
export default function FlythroughController({
  active = false,
  isPlaying = true,
  speed = 1.0,
  controlsRef = null,
  onProgressUpdate = null,
}) {
  const { camera } = useThree();
  const progressRef = useRef(0.0);
  const prevActiveRef = useRef(active);

  // Build Catmull-Rom spline curve
  const { curve, pathPoints } = useMemo(() => {
    const c = new THREE.CatmullRomCurve3(SPLINE_WAYPOINTS, false, 'catmullrom', 0.4);
    const pts = c.getPoints(100).map(p => [p.x, p.y, p.z]);
    return { curve: c, pathPoints: pts };
  }, []);

  // Handle transitions into/out of flythrough mode
  useEffect(() => {
    if (controlsRef?.current) {
      controlsRef.current.enabled = !active;
    }

    if (active && !prevActiveRef.current) {
      // Reset flight to beginning on activation
      progressRef.current = 0.0;
    }

    prevActiveRef.current = active;

    return () => {
      if (controlsRef?.current) {
        controlsRef.current.enabled = true;
      }
    };
  }, [active, controlsRef]);

  // Frame animation loop
  useFrame((state, delta) => {
    if (!active) return;

    // Advance flight along spline
    if (isPlaying) {
      const step = delta * 0.035 * Math.max(0.2, speed);
      progressRef.current = (progressRef.current + step) % 1.0;
    }

    const t = progressRef.current;
    const cameraPos = curve.getPointAt(t);

    // Look-ahead target (look slightly forward along flight path)
    const lookAheadT = (t + 0.04) % 1.0;
    const lookAtTarget = curve.getPointAt(lookAheadT);

    // Smoothly glide camera
    state.camera.position.lerp(cameraPos, 0.15);
    state.camera.lookAt(lookAtTarget);

    if (onProgressUpdate) {
      onProgressUpdate({
        progress: t,
        altitude: cameraPos.y,
        position: cameraPos,
      });
    }
  });

  if (!active) return null;

  return (
    <group>
      {/* Visual 3D flight path ribbon */}
      <Line
        points={pathPoints}
        color="#00e5ff"
        lineWidth={2}
        transparent
        opacity={0.5}
        dashed
        dashScale={2}
        dashSize={1}
        gapSize={0.5}
      />
    </group>
  );
}
