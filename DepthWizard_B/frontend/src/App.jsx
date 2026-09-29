import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Satellite,
  Mountain,
  Layers,
  Ruler,
  Plane,
  Play,
  Pause,
  Download,
  Upload,
  RefreshCw,
  Sliders,
  Eye,
  Check,
  AlertCircle,
  ChevronUp,
  ChevronDown,
  Droplets,
  Trash2,
  Maximize2,
  Activity,
  ShieldCheck,
} from 'lucide-react';

import {
  fetchPresets,
  reconstructTerrain,
  decodeHeightMatrix,
  getPresetImageUrl,
  exportGeoTIFF,
  exportGLB,
} from './lib/api';

import Canvas3D from './components/Canvas3D';
import ValidationHUD from './components/ValidationHUD';
import TopoCrossSection from './components/TopoCrossSection';
import DiagnosticDrawer from './components/DiagnosticDrawer';

export default function App() {
  // ── Regional Presets & Reconstruction State ────────────────────────────────
  const [presets, setPresets] = useState([]);
  const [selectedPresetId, setSelectedPresetId] = useState('leh');
  const [uploadedFile, setUploadedFile] = useState(null);
  const [uploadedPreviewUrl, setUploadedPreviewUrl] = useState(null);
  const [waterLeveling, setWaterLeveling] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Active Reconstruction & 3D DSM Data
  const [reconstruction, setReconstruction] = useState(null);
  const [heightData, setHeightData] = useState(null);
  const [activeOpticalUrl, setActiveOpticalUrl] = useState(null);

  // ── 3D Viewport Controls ───────────────────────────────────────────────────
  const [verticalExaggeration, setVerticalExaggeration] = useState(1.0);
  const [colorMode, setColorMode] = useState('optical'); // 'optical' | 'hypsometric' | 'wireframe'

  // ── Elevation Ruler State ──────────────────────────────────────────────────
  const [rulerActive, setRulerActive] = useState(false);
  const [pinA, setPinA] = useState(null);
  const [pinB, setPinB] = useState(null);
  const [crossSectionProfile, setCrossSectionProfile] = useState(null);
  const [profileDrawerOpen, setProfileDrawerOpen] = useState(true);

  // ── Reconnaissance Flythrough State ────────────────────────────────────────
  const [flythroughActive, setFlythroughActive] = useState(false);
  const [flythroughPlaying, setFlythroughPlaying] = useState(true);
  const [flythroughSpeed, setFlythroughSpeed] = useState(1.0);
  const [flythroughTelemetry, setFlythroughTelemetry] = useState(null);

  // ── Export State ───────────────────────────────────────────────────────────
  const [exportingTiff, setExportingTiff] = useState(false);
  const [exportingGlb, setExportingGlb] = useState(false);

  const fileInputRef = useRef(null);

  // ── Load Presets on Mount ──────────────────────────────────────────────────
  useEffect(() => {
    async function initPresets() {
      try {
        const list = await fetchPresets();
        setPresets(list);
        if (list.length > 0) {
          const defaultPreset = list.find((p) => p.id === 'leh') || list[0];
          setSelectedPresetId(defaultPreset.id);
          triggerReconstruction({ presetId: defaultPreset.id });
        }
      } catch (err) {
        console.warn('Initial preset load warning:', err);
        // If backend is warming up or presets empty, attempt reconstruction anyway
        triggerReconstruction({ presetId: 'leh' });
      }
    }
    initPresets();
  }, []);

  // ── Core Reconstruction Trigger ────────────────────────────────────────────
  const triggerReconstruction = useCallback(
    async ({ presetId = null, file = null } = {}) => {
      setLoading(true);
      setError(null);

      // Clear any prior ruler pins
      setPinA(null);
      setPinB(null);
      setCrossSectionProfile(null);

      try {
        const targetPreset = file ? null : (presetId || selectedPresetId);
        const targetFile = file || (targetPreset ? null : uploadedFile);

        const result = await reconstructTerrain({
          preset_id: targetPreset,
          image_file: targetFile,
          level_water: waterLeveling,
        });

        setReconstruction(result);

        // Decode binary height matrix
        const decoded = decodeHeightMatrix(result.height_matrix);
        setHeightData(decoded);

        // Set optical texture URL
        if (targetPreset) {
          setActiveOpticalUrl(getPresetImageUrl(targetPreset));
        } else if (targetFile) {
          const objectUrl = URL.createObjectURL(targetFile);
          setActiveOpticalUrl(objectUrl);
        }
      } catch (err) {
        console.error('Reconstruction error:', err);
        setError(err.message || 'Reconstruction failed. Please check backend connection.');
      } finally {
        setLoading(false);
      }
    },
    [selectedPresetId, uploadedFile, waterLeveling]
  );

  // ── Custom Image Upload Handler ────────────────────────────────────────────
  const handleFileUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploadedFile(file);
    setSelectedPresetId(null);

    const preview = URL.createObjectURL(file);
    setUploadedPreviewUrl(preview);

    // Auto-trigger reconstruction for uploaded image
    triggerReconstruction({ file });
  };

  // ── Reset Elevation Ruler ──────────────────────────────────────────────────
  const handleResetRuler = () => {
    setPinA(null);
    setPinB(null);
    setCrossSectionProfile(null);
  };

  // ── Export Handlers ────────────────────────────────────────────────────────
  const handleExportGeoTIFF = async () => {
    if (!reconstruction?.height_matrix) return;
    try {
      setExportingTiff(true);
      await exportGeoTIFF({
        preset_id: reconstruction.preset_id,
        height_matrix: reconstruction.height_matrix,
      });
    } catch (err) {
      alert(`Export GeoTIFF failed: ${err.message}`);
    } finally {
      setExportingTiff(false);
    }
  };

  const handleExportGLB = async () => {
    if (!reconstruction?.height_matrix) return;
    try {
      setExportingGlb(true);
      await exportGLB({
        preset_id: reconstruction.preset_id,
        height_matrix: reconstruction.height_matrix,
        vertical_exaggeration: verticalExaggeration,
      });
    } catch (err) {
      alert(`Export GLB failed: ${err.message}`);
    } finally {
      setExportingGlb(false);
    }
  };

  return (
    <div className="flex h-screen w-screen bg-[#030712] text-slate-100 font-sans overflow-hidden select-none">
      {/* ──────────────────────────────────────────────────────────────────────
          LEFT SIDEBAR: CONTROLS & TELEMETRY
      ────────────────────────────────────────────────────────────────────── */}
      <aside className="w-80 sm:w-96 flex-shrink-0 bg-slate-950/95 border-r border-slate-800/80 flex flex-col z-30 shadow-2xl backdrop-blur-md overflow-y-auto">
        {/* Header Branding */}
        <div className="p-4 border-b border-slate-800/90 bg-gradient-to-b from-slate-900/80 to-transparent">
          <div className="flex items-center justify-between mb-1.5">
            <div className="flex items-center gap-2">
              <div className="p-1.5 rounded-lg bg-cyan-500/20 text-cyan-400 border border-cyan-500/30">
                <Satellite className="w-5 h-5 animate-pulse-subtle" />
              </div>
              <div>
                <h1 className="text-sm font-bold tracking-wider font-mono uppercase text-white">
                  DepthWizard
                </h1>
                <p className="text-[10px] text-slate-400 font-mono">
                  Single-View 3D DSM
                </p>
              </div>
            </div>
            <div className="px-2 py-0.5 rounded text-[10px] font-mono bg-emerald-950/70 border border-emerald-500/40 text-emerald-400">
              OFFLINE AI
            </div>
          </div>
        </div>

        {/* Control Panels */}
        <div className="p-4 space-y-5 flex-1 text-xs">
          {/* Section: AOI & Presets */}
          <div className="space-y-2">
            <div className="mb-2">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold tracking-wider text-cyan-400 flex items-center gap-1">
                  <Mountain className="w-3.5 h-3.5 text-cyan-400" />
                  TARGET REGION (AOI)
                </span>
                <span className="text-[10px] text-slate-500 font-mono">Tactical Preset</span>
              </div>
              <div className="text-[10px] text-slate-400 font-mono tracking-tight mt-0.5">
                (for Georeferenced RGB Imagery)
              </div>
            </div>

            {/* Regional Preset Buttons */}
            <div className="grid grid-cols-3 gap-1.5">
              {[
                { id: 'leh', label: 'Leh, Ladakh', elev: '3500-5300m', badge: 'High-Alt' },
                { id: 'hyderabad', label: 'Hyderabad', elev: '540-660m', badge: 'Plateau' },
                { id: 'assam', label: 'Assam Valley', elev: '120-160m', badge: 'Floodplain' },
              ].map((p) => {
                const isSelected = selectedPresetId === p.id;
                return (
                  <button
                    key={p.id}
                    onClick={() => {
                      setSelectedPresetId(p.id);
                      setUploadedFile(null);
                      triggerReconstruction({ presetId: p.id });
                    }}
                    className={`p-2 rounded border text-left transition-all ${
                      isSelected
                        ? 'bg-cyan-950/70 border-cyan-500 text-cyan-300 shadow-hud'
                        : 'bg-slate-900/60 border-slate-800 text-slate-400 hover:border-slate-700 hover:text-slate-200'
                    }`}
                  >
                    <div className="font-bold font-mono text-[11px] truncate">{p.label}</div>
                    <div className="text-[9px] text-slate-500 font-mono truncate">{p.elev}</div>
                    <span className="inline-block mt-1 text-[8px] px-1 py-0.2 rounded bg-slate-800/80 text-slate-400 uppercase font-mono">
                      {p.badge}
                    </span>
                  </button>
                );
              })}
            </div>

            {/* Subtle horizontal rule separating presets from custom uploads */}
            <div className="my-3 border-t border-slate-700/60" />

            {/* Custom Image Upload */}
            <div className="pt-0.5">
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*,.tif,.tiff"
                onChange={handleFileUpload}
                className="hidden"
              />
              <button
                onClick={() => fileInputRef.current?.click()}
                className={`w-full py-2.5 px-3 rounded border transition-all flex flex-col items-center justify-center gap-0.5 ${
                  uploadedFile
                    ? 'bg-amber-950/60 border-amber-500/60 text-amber-300'
                    : 'border-dashed border-slate-700 hover:border-cyan-500/50 bg-slate-900/40 hover:bg-slate-800/60 text-slate-300'
                }`}
              >
                <div className="flex items-center gap-2 text-xs font-medium">
                  <Upload className="w-3.5 h-3.5 text-cyan-400" />
                  <span className="truncate">
                    {uploadedFile ? `Custom: ${uploadedFile.name}` : 'Upload Recon Optical Image'}
                  </span>
                </div>
                <span className="text-[10px] text-slate-400 font-mono">
                  (for Non-Georeferenced RGB Imagery)
                </span>
              </button>
            </div>
          </div>

          {/* Section: Pipeline Options */}
          <div className="space-y-3 pt-2 border-t border-slate-800/80">
            <div className="flex items-center justify-between text-slate-400 font-mono text-[11px]">
              <span className="uppercase tracking-wider flex items-center gap-1">
                <Sliders className="w-3.5 h-3.5 text-cyan-400" />
                Processing Pipeline
              </span>
            </div>

            {/* Shadow-safe Water Leveling Toggle */}
            <label className="flex items-center justify-between p-2 rounded bg-slate-900/60 border border-slate-800 cursor-pointer hover:bg-slate-900/80 transition-colors">
              <div className="flex items-center gap-2">
                <Droplets className="w-3.5 h-3.5 text-cyan-400" />
                <div>
                  <div className="font-semibold text-slate-200">Shadow-Safe Water Leveling</div>
                  <div className="text-[10px] text-slate-500">VWI + Otsu + Luminance Gate</div>
                </div>
              </div>
              <input
                type="checkbox"
                checked={waterLeveling}
                onChange={(e) => setWaterLeveling(e.target.checked)}
                className="w-4 h-4 rounded bg-slate-800 border-slate-700 text-cyan-500 focus:ring-0 focus:ring-offset-0 cursor-pointer"
              />
            </label>

            {/* Reconstruct Trigger Button */}
            <button
              disabled={loading}
              onClick={() => triggerReconstruction()}
              className="w-full py-2.5 px-4 rounded bg-cyan-600 hover:bg-cyan-500 active:bg-cyan-700 text-slate-950 font-bold font-mono text-xs flex items-center justify-center gap-2 transition-all shadow-hud disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
              <span>{loading ? 'RECONSTRUCTING 3D DSM...' : 'RUN 3D RECONSTRUCTION'}</span>
            </button>

            {error && (
              <div className="p-2.5 rounded bg-rose-950/60 border border-rose-500/50 text-rose-300 text-[11px] font-mono flex items-start gap-2">
                <AlertCircle className="w-4 h-4 text-rose-400 flex-shrink-0 mt-0.5" />
                <div className="truncate">{error}</div>
              </div>
            )}
          </div>

          {/* Section: 3D Visualization Modes */}
          <div className="space-y-3 pt-2 border-t border-slate-800/80">
            <div className="flex items-center justify-between text-slate-400 font-mono text-[11px]">
              <span className="uppercase tracking-wider flex items-center gap-1">
                <Layers className="w-3.5 h-3.5 text-cyan-400" />
                Surface Shader
              </span>
            </div>

            {/* Color Mode Switch */}
            <div className="grid grid-cols-3 gap-1 p-0.5 bg-slate-900 border border-slate-800 rounded">
              {[
                { id: 'optical', label: 'Optical (RGB)' },
                { id: 'hypsometric', label: 'Heatmap (DEM)' },
                { id: 'wireframe', label: 'Wireframe' },
              ].map((m) => (
                <button
                  key={m.id}
                  onClick={() => setColorMode(m.id)}
                  className={`py-1.5 px-0.5 text-center font-mono font-semibold text-[10px] leading-tight rounded transition-all ${
                    colorMode === m.id
                      ? 'bg-cyan-500 text-slate-950 shadow-sm'
                      : 'text-slate-400 hover:text-white'
                  }`}
                >
                  {m.label}
                </button>
              ))}
            </div>

            {/* Vertical Exaggeration Slider */}
            <div className="space-y-1">
              <div className="flex justify-between text-[11px] font-mono">
                <span className="text-slate-400">Vertical Relief Exaggeration</span>
                <span className="text-cyan-400 font-bold">{verticalExaggeration.toFixed(1)}x</span>
              </div>
              <input
                type="range"
                min="0.5"
                max="3.0"
                step="0.1"
                value={verticalExaggeration}
                onChange={(e) => setVerticalExaggeration(parseFloat(e.target.value))}
                className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-400"
              />
              <div className="flex justify-between text-[9px] text-slate-500 font-mono">
                <span>0.5x (Subtle)</span>
                <span>1.0x (Standard)</span>
                <span>3.0x (Extreme)</span>
              </div>
            </div>
          </div>

          {/* Section: Tactical Interactive Tools */}
          <div className="space-y-3 pt-2 border-t border-slate-800/80">
            <div className="flex items-center justify-between text-slate-400 font-mono text-[11px]">
              <span className="uppercase tracking-wider flex items-center gap-1">
                <Ruler className="w-3.5 h-3.5 text-cyan-400" />
                Tactical Tools
              </span>
            </div>

            {/* Elevation Ruler Toggle */}
            <div className="p-2.5 rounded bg-slate-900/60 border border-slate-800 space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Ruler className="w-4 h-4 text-cyan-400" />
                  <span className="font-semibold text-slate-200">Elevation Ruler</span>
                </div>
                <button
                  onClick={() => {
                    const next = !rulerActive;
                    setRulerActive(next);
                    if (next) setFlythroughActive(false); // mutually exclusive with flythrough
                  }}
                  className={`px-2.5 py-1 rounded text-[10px] font-mono font-bold transition-all ${
                    rulerActive
                      ? 'bg-cyan-500 text-slate-950'
                      : 'bg-slate-800 text-slate-400 hover:text-white'
                  }`}
                >
                  {rulerActive ? 'ARMED' : 'STANDBY'}
                </button>
              </div>

              {rulerActive && (
                <div className="text-[10px] text-slate-400 font-mono space-y-1.5 pt-1 border-t border-slate-800/80">
                  <div className="flex justify-between items-center">
                    <span>Pin A: {pinA ? `${pinA.elev.toFixed(1)}m` : 'Not set'}</span>
                    <span>Pin B: {pinB ? `${pinB.elev.toFixed(1)}m` : 'Not set'}</span>
                  </div>
                  <button
                    onClick={handleResetRuler}
                    className="w-full py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-300 flex items-center justify-center gap-1 transition-colors"
                  >
                    <Trash2 className="w-3 h-3 text-rose-400" />
                    <span>Clear Measured Pins</span>
                  </button>
                </div>
              )}
            </div>

            {/* Recon Flythrough Toggle */}
            <div className="p-2.5 rounded bg-slate-900/60 border border-slate-800 space-y-2">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Plane className="w-4 h-4 text-cyan-400" />
                  <span className="font-semibold text-slate-200">Recon Flythrough</span>
                </div>
                <button
                  onClick={() => {
                    const next = !flythroughActive;
                    setFlythroughActive(next);
                    if (next) setRulerActive(false);
                  }}
                  className={`px-2.5 py-1 rounded text-[10px] font-mono font-bold transition-all ${
                    flythroughActive
                      ? 'bg-emerald-500 text-slate-950'
                      : 'bg-slate-800 text-slate-400 hover:text-white'
                  }`}
                >
                  {flythroughActive ? 'ACTIVE' : 'OFF'}
                </button>
              </div>

              {flythroughActive && (
                <div className="space-y-2 pt-1 border-t border-slate-800/80">
                  <div className="flex items-center justify-between text-[11px] font-mono">
                    <button
                      onClick={() => setFlythroughPlaying(!flythroughPlaying)}
                      className="px-2 py-1 rounded bg-slate-800 hover:bg-slate-700 text-slate-200 flex items-center gap-1"
                    >
                      {flythroughPlaying ? <Pause className="w-3 h-3 text-amber-400" /> : <Play className="w-3 h-3 text-emerald-400" />}
                      <span>{flythroughPlaying ? 'Pause' : 'Resume'}</span>
                    </button>
                    <span className="text-slate-400">
                      Speed: <strong className="text-cyan-400">{flythroughSpeed.toFixed(1)}x</strong>
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0.5"
                    max="3.0"
                    step="0.25"
                    value={flythroughSpeed}
                    onChange={(e) => setFlythroughSpeed(parseFloat(e.target.value))}
                    className="w-full h-1 bg-slate-800 rounded appearance-none cursor-pointer accent-cyan-400"
                  />
                </div>
              )}
            </div>
          </div>

          {/* Section: Export Formats */}
          <div className="space-y-2 pt-2 border-t border-slate-800/80">
            <div className="text-slate-400 font-mono text-[11px] uppercase tracking-wider flex items-center gap-1">
              <Download className="w-3.5 h-3.5 text-cyan-400" />
              Tactical Deliverables
            </div>
            <div className="grid grid-cols-2 gap-2">
              <button
                disabled={!reconstruction || exportingTiff}
                onClick={handleExportGeoTIFF}
                className="py-2 px-2.5 rounded bg-slate-900 border border-slate-800 hover:border-slate-700 text-slate-200 font-mono text-[11px] font-semibold flex items-center justify-center gap-1.5 transition-all disabled:opacity-40"
              >
                <Download className="w-3.5 h-3.5 text-cyan-400" />
                <span>{exportingTiff ? 'Exporting...' : 'GeoTIFF (DEM)'}</span>
              </button>

              <button
                disabled={!reconstruction || exportingGlb}
                onClick={handleExportGLB}
                className="py-2 px-2.5 rounded bg-slate-900 border border-slate-800 hover:border-slate-700 text-slate-200 font-mono text-[11px] font-semibold flex items-center justify-center gap-1.5 transition-all disabled:opacity-40"
              >
                <Download className="w-3.5 h-3.5 text-amber-400" />
                <span>{exportingGlb ? 'Exporting...' : '3D Mesh (GLB)'}</span>
              </button>
            </div>
          </div>
        </div>

        {/* Footer Telemetry */}
        <div className="p-3 border-t border-slate-800 bg-slate-950 text-[10px] font-mono text-slate-500 flex items-center justify-between">
          <span>Grid: 256×256 px</span>
          <span className="text-cyan-400 font-semibold">
            {reconstruction?.mode === 'tactical' ? 'CALIBRATED AMSL' : 'RELATIVE RELIEF'}
          </span>
        </div>
      </aside>

      {/* ──────────────────────────────────────────────────────────────────────
          CENTRAL 3D VIEWPORT & FLOATING HUD OVERLAYS
      ────────────────────────────────────────────────────────────────────── */}
      <main className="flex-1 relative h-full w-full overflow-hidden">
        {/* Fullscreen 3D Canvas */}
        <Canvas3D
          heightData={heightData}
          opticalUrl={activeOpticalUrl}
          verticalExaggeration={verticalExaggeration}
          colorMode={colorMode}
          onColorModeChange={setColorMode}
          groundResolution={reconstruction?.ground_resolution_m_per_px}
          units={reconstruction?.units || 'meters_amsl'}
          mode={reconstruction?.mode || 'relative'}
          rulerActive={rulerActive}
          onProfileGenerated={setCrossSectionProfile}
          flythroughActive={flythroughActive}
          flythroughPlaying={flythroughPlaying}
          flythroughSpeed={flythroughSpeed}
          onFlythroughProgress={setFlythroughTelemetry}
          pinA={pinA}
          pinB={pinB}
          setPinA={setPinA}
          setPinB={setPinB}
        />

        {/* Top-Right Floating Validation HUD */}
        <div className="absolute top-4 right-4 z-20 pointer-events-none flex flex-col items-end">
          <ValidationHUD
            mode={reconstruction?.mode || 'relative'}
            calibration={reconstruction?.calibration}
            metrics={reconstruction?.metrics}
            groundResolution={reconstruction?.ground_resolution_m_per_px}
            presetId={reconstruction?.preset_id}
          />
        </div>

        {/* Bottom Expandable Topo Cross-Section Drawer */}
        <div className="absolute bottom-4 left-4 right-4 sm:left-8 sm:right-8 z-20 pointer-events-auto max-w-4xl mx-auto">
          <div className="relative">
            {/* Drawer Header & Expand/Collapse Toggle */}
            <div className="flex justify-end mb-1">
              <button
                onClick={() => setProfileDrawerOpen(!profileDrawerOpen)}
                className="px-2.5 py-1 rounded-t bg-slate-950/90 border border-b-0 border-slate-800 text-slate-300 font-mono text-[10px] flex items-center gap-1 backdrop-blur-md shadow-md hover:text-white"
              >
                <span>Elevation Cross-Section</span>
                {profileDrawerOpen ? <ChevronDown className="w-3 h-3" /> : <ChevronUp className="w-3 h-3" />}
              </button>
            </div>

            {/* Collapsible Chart Body */}
            {profileDrawerOpen && (
              <TopoCrossSection
                profile={crossSectionProfile}
                units={reconstruction?.units || 'meters_amsl'}
                mode={reconstruction?.mode || 'relative'}
              />
            )}
          </div>
        </div>

        {/* Flythrough Telemetry HUD Badge */}
        {flythroughActive && flythroughTelemetry && (
          <div className="absolute top-4 left-4 z-20 pointer-events-none">
            <div className="p-3 rounded-lg bg-slate-950/90 border border-cyan-500/40 text-cyan-300 font-mono text-xs shadow-hud-glow backdrop-blur-md space-y-1">
              <div className="flex items-center gap-2 font-bold text-white border-b border-slate-800 pb-1">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-ping"></span>
                <span>RECON DRONE CAM</span>
                <span className="text-[10px] text-slate-400">CORRIDOR 01</span>
              </div>
              <div className="grid grid-cols-2 gap-x-3 text-[11px]">
                <span className="text-slate-400">Altitude:</span>
                <span className="text-right font-bold text-white">
                  {(flythroughTelemetry.altitude * 100).toFixed(0)} m
                </span>
                <span className="text-slate-400">Progress:</span>
                <span className="text-right font-bold text-cyan-400">
                  {(flythroughTelemetry.progress * 100).toFixed(0)}%
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Bottom-Left: Pipeline Diagnostics Drawer */}
        <div className="absolute bottom-4 left-4 z-20 w-72 sm:w-80 pointer-events-auto">
          <DiagnosticDrawer
            opticalUrl={activeOpticalUrl}
            heightData={heightData}
            waterMaskApplied={reconstruction?.water_mask_applied ?? false}
            reconstruction={reconstruction}
          />
        </div>
      </main>
    </div>
  );
}
