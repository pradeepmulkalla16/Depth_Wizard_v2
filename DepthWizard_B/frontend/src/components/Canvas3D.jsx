import React, { useRef, useState, useCallback } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, Center } from '@react-three/drei';
import TerrainMesh from './TerrainMesh';
import ElevationRuler, { extractElevationIntersection } from './ElevationRuler';
import FlythroughController from './FlythroughController';

/**
 * Canvas3D
 * ────────────────────────────────────────────────────────────────────────────
 * Tactical 3D Viewport container.
 * 
 * Embeds:
 *   - Atmospheric background & fog
 *   - Solar azimuth directional lighting + ambient fill
 *   - Damped OrbitControls (disabled during flythrough)
 *   - Displaced TerrainMesh (DSM)
 *   - Interactive Raycasting ElevationRuler
 *   - Catmull-Rom FlythroughController
 * ────────────────────────────────────────────────────────────────────────────
 */
export default function Canvas3D({
  heightData,
  opticalUrl = null,
  verticalExaggeration = 1.0,
  colorMode = 'optical',
  onColorModeChange = null,
  groundResolution = null,
  units = 'meters_amsl',
  mode = 'tactical',
  rulerActive = false,
  onProfileGenerated = null,
  flythroughActive = false,
  flythroughPlaying = true,
  flythroughSpeed = 1.0,
  onFlythroughProgress = null,
  pinA = null,
  pinB = null,
  setPinA = null,
  setPinB = null,
}) {
  const controlsRef = useRef(null);

  // Handle click on terrain mesh for raycast pin placement
  const handleTerrainClick = useCallback((e) => {
    // If ruler is not active or flythrough is running, don't drop pins
    if (!rulerActive || flythroughActive) return;

    e.stopPropagation();

    const intersection = e.intersections && e.intersections.length > 0 ? e.intersections[0] : null;
    if (!intersection) return;

    const hit = extractElevationIntersection(intersection, heightData);
    if (!hit) return;

    if (!pinA) {
      if (setPinA) setPinA(hit);
    } else if (pinA && !pinB) {
      if (setPinB) setPinB(hit);
    } else {
      // Both already set: start fresh with Pin A at new location
      if (setPinA) setPinA(hit);
      if (setPinB) setPinB(null);
    }
  }, [rulerActive, flythroughActive, heightData, pinA, pinB, setPinA, setPinB]);

  return (
    <div className="w-full h-full relative select-none bg-[#030712] overflow-hidden">
      <Canvas
        camera={{ position: [0, 65, 85], fov: 45, near: 0.5, far: 1000 }}
        shadows
        gl={{ antialias: true, alpha: false, powerPreference: 'high-performance' }}
      >
        {/* Deep space / atmospheric backdrop */}
        <color attach="background" args={['#030712']} />
        <fog attach="fog" args={['#030712', 80, 350]} />

        {/* Tactical Solar Lighting */}
        <ambientLight intensity={0.45} color="#94a3b8" />

        {/* Primary Sun Light (simulating sun angle & cast shadows across ridges) */}
        <directionalLight
          position={[60, 90, 45]}
          intensity={1.9}
          castShadow
          shadow-mapSize-width={2048}
          shadow-mapSize-height={2048}
          shadow-camera-near={10}
          shadow-camera-far={300}
          shadow-camera-left={-70}
          shadow-camera-right={70}
          shadow-camera-top={70}
          shadow-camera-bottom={-70}
          shadow-bias={-0.0005}
        />

        {/* Secondary Fill / Rim Light */}
        <directionalLight
          position={[-50, 30, -50]}
          intensity={0.4}
          color="#38bdf8"
        />

        {/* Low-angle architectural rake light — grazes across vertical building faces
            so walls are brightly lit and rooftops/ground remain dim, separating merged mounds */}
        <directionalLight
          position={[0, 15, 80]}
          intensity={0.55}
          color="#e2e8f0"
        />

        {/* Tactical Base Reference Grid */}
        <gridHelper
          args={[140, 28, '#00e5ff', '#1e293b']}
          position={[0, -0.2, 0]}
        />

        {/* Orbit Camera Controls */}
        <OrbitControls
          ref={controlsRef}
          makeDefault
          enableDamping
          dampingFactor={0.06}
          maxPolarAngle={Math.PI / 2 - 0.05}
          minDistance={10}
          maxDistance={350}
          target={[0, 0, 0]}
        />

        {/* 3D Terrain Surface Mesh */}
        {heightData && (
          <TerrainMesh
            heightData={heightData}
            opticalUrl={opticalUrl}
            verticalExaggeration={verticalExaggeration}
            colorMode={colorMode}
            onPointerDown={handleTerrainClick}
          />
        )}

        {/* Point-to-Point Elevation Ruler in 3D */}
        <ElevationRuler
          heightData={heightData}
          groundResolution={groundResolution}
          units={units}
          mode={mode}
          active={rulerActive}
          onProfileGenerated={onProfileGenerated}
          pinA={pinA}
          pinB={pinB}
          setPinA={setPinA}
          setPinB={setPinB}
        />

        {/* Reconnaissance Drone Flythrough Controller */}
        <FlythroughController
          active={flythroughActive}
          isPlaying={flythroughPlaying}
          speed={flythroughSpeed}
          controlsRef={controlsRef}
          onProgressUpdate={onFlythroughProgress}
        />
      </Canvas>

      {/* Crosshair cursor indicator when ruler is active */}
      {rulerActive && !flythroughActive && (
        <div className="absolute top-14 left-1/2 -translate-x-1/2 z-20 pointer-events-none">
          <div className="px-3 py-1.5 rounded-full bg-slate-950/85 border border-cyan-500/50 text-cyan-300 font-mono text-xs shadow-hud-glow backdrop-blur-md flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping"></span>
            <span>
              {!pinA ? 'CLICK TERRAIN TO SET PIN A' : !pinB ? 'CLICK TERRAIN TO SET PIN B' : 'CLICK TO REPOSITION PIN A'}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
