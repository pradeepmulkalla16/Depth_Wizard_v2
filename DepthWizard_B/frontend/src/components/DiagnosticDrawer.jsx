import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  FlaskConical,
  ChevronUp,
  ChevronDown,
  Image,
  Layers,
  Droplets,
  X,
  ZoomIn,
  CheckCircle2,
  AlertTriangle,
} from 'lucide-react';

// ─────────────────────────────────────────────────────────────────────────────
// Magma colormap — 8 evenly-spaced stops from deep purple to bright yellow-white
// Mirrors matplotlib "magma" so the depth preview matches scientific convention.
// ─────────────────────────────────────────────────────────────────────────────
const MAGMA_STOPS = [
  [0,   0,  4],    // 0.00 – near black / deepest
  [28,  16,  68],  // 0.14 – dark purple
  [79,  18, 123],  // 0.28 – purple-violet
  [129,  37, 129], // 0.42 – mid violet
  [181,  54, 122], // 0.57 – rose-magenta
  [229,  80,  100], // 0.71 – coral
  [251, 135,  97], // 0.85 – peach
  [252, 253, 191], // 1.00 – pale yellow-white / shallowest
];

function sampleMagma(t) {
  const clamped = Math.max(0, Math.min(1, t));
  const n = MAGMA_STOPS.length - 1;
  const scaledIdx = clamped * n;
  const lo = Math.floor(scaledIdx);
  const hi = Math.min(lo + 1, n);
  const alpha = scaledIdx - lo;

  const [r0, g0, b0] = MAGMA_STOPS[lo];
  const [r1, g1, b1] = MAGMA_STOPS[hi];

  return [
    Math.round(r0 + (r1 - r0) * alpha),
    Math.round(g0 + (g1 - g0) * alpha),
    Math.round(b0 + (b1 - b0) * alpha),
  ];
}

// ─────────────────────────────────────────────────────────────────────────────
// Off-screen canvas painter — converts Float32Array heightData → magma RGBA PNG
// ─────────────────────────────────────────────────────────────────────────────
function paintDepthToCanvas(canvasEl, heightData) {
  if (!canvasEl || !heightData?.data) return;

  const { width = 256, height = 256, data, minElev, maxElev } = heightData;
  const range = Math.max((maxElev ?? 100) - (minElev ?? 0), 1e-6);

  canvasEl.width = width;
  canvasEl.height = height;

  const ctx = canvasEl.getContext('2d');
  const imageData = ctx.createImageData(width, height);
  const pixels = imageData.data;

  for (let i = 0; i < data.length; i++) {
    const t = (data[i] - (minElev ?? 0)) / range;
    const [r, g, b] = sampleMagma(t);
    pixels[i * 4 + 0] = r;
    pixels[i * 4 + 1] = g;
    pixels[i * 4 + 2] = b;
    pixels[i * 4 + 3] = 255;
  }

  ctx.putImageData(imageData, 0, 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// Lightbox — fullscreen modal to inspect any thumbnail
// ─────────────────────────────────────────────────────────────────────────────
function Lightbox({ title, children, onClose }) {
  // Close on Escape
  useEffect(() => {
    const handler = (e) => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/90 backdrop-blur-sm"
      onClick={onClose}
      role="dialog"
      aria-modal="true"
      aria-label={title}
    >
      <div
        className="relative max-w-3xl w-full mx-4 bg-slate-900 border border-cyan-500/40 rounded-xl shadow-[0_0_60px_rgba(6,182,212,0.2)] overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Modal Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800">
          <span className="font-mono text-xs font-bold uppercase tracking-widest text-cyan-400">
            {title}
          </span>
          <button
            id="lightbox-close-btn"
            onClick={onClose}
            className="p-1 rounded-full bg-slate-800 hover:bg-slate-700 text-slate-400 hover:text-white transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-4 flex items-center justify-center bg-slate-950/40 min-h-[320px]">
          {children}
        </div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Individual thumbnail card
// ─────────────────────────────────────────────────────────────────────────────
function DiagCard({ id, title, subtitle, badge, badgeColor = 'cyan', children, lightboxContent }) {
  const [lightboxOpen, setLightboxOpen] = useState(false);

  const badgeColors = {
    cyan: 'bg-cyan-950/80 border-cyan-500/50 text-cyan-300',
    emerald: 'bg-emerald-950/80 border-emerald-500/50 text-emerald-300',
    amber: 'bg-amber-950/80 border-amber-500/50 text-amber-300',
    slate: 'bg-slate-800/80 border-slate-600/50 text-slate-300',
  };

  return (
    <>
      <div
        id={id}
        className="flex-1 min-w-0 bg-slate-950/70 border border-slate-800/80 rounded-lg overflow-hidden group cursor-pointer hover:border-cyan-500/50 transition-all hover:shadow-[0_0_12px_rgba(6,182,212,0.15)]"
        onClick={() => setLightboxOpen(true)}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => e.key === 'Enter' && setLightboxOpen(true)}
        aria-label={`Inspect ${title}`}
      >
        {/* Thumbnail area */}
        <div className="relative w-full aspect-square bg-slate-900/80 overflow-hidden">
          {children}
          {/* Zoom indicator on hover */}
          <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity bg-slate-950/40">
            <ZoomIn className="w-6 h-6 text-cyan-400 drop-shadow-lg" />
          </div>
        </div>

        {/* Card footer */}
        <div className="p-2 space-y-1">
          <div className="font-mono text-[10px] font-bold text-slate-200 truncate">{title}</div>
          <div className="text-[9px] text-slate-500 truncate">{subtitle}</div>
          {badge && (
            <span className={`inline-block text-[8px] px-1.5 py-0.5 rounded border font-mono uppercase font-bold ${badgeColors[badgeColor]}`}>
              {badge}
            </span>
          )}
        </div>
      </div>

      {lightboxOpen && (
        <Lightbox title={title} onClose={() => setLightboxOpen(false)}>
          {lightboxContent || children}
        </Lightbox>
      )}
    </>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// DiagnosticDrawer — main export
// ─────────────────────────────────────────────────────────────────────────────

/**
 * DiagnosticDrawer
 * ────────────────────────────────────────────────────────────────────────────
 * Collapsible 2D pipeline diagnostics panel anchored at the bottom-left of
 * the 3D viewport. Shows three side-by-side thumbnail cards:
 *   1. Input Optical RGB — source satellite tile
 *   2. Predicted Depth — magma colormap synthesized from heightData Float32Array
 *   3. Water Sanitization Mask — badge/overlay based on waterMaskApplied flag
 *
 * Client-side synthesis means zero backend changes and full offline operation.
 *
 * @param {string|null}   opticalUrl       - URL of the optical satellite tile
 * @param {Object|null}   heightData       - Decoded {data, width, height, minElev, maxElev}
 * @param {boolean}       waterMaskApplied - Whether water leveling was applied in this run
 * @param {Object|null}   reconstruction   - Full reconstruction payload from backend
 */
export default function DiagnosticDrawer({
  opticalUrl = null,
  heightData = null,
  waterMaskApplied = false,
  reconstruction = null,
}) {
  const [open, setOpen] = useState(false);

  // Canvas ref for synthetic depth preview
  const depthCanvasRef = useRef(null);
  const depthLightboxCanvasRef = useRef(null);

  // Paint depth canvas whenever heightData changes
  const paintDepth = useCallback(() => {
    paintDepthToCanvas(depthCanvasRef.current, heightData);
    paintDepthToCanvas(depthLightboxCanvasRef.current, heightData);
  }, [heightData]);

  useEffect(() => {
    if (open) {
      // Paint on next tick so canvas DOM is mounted
      const timer = setTimeout(paintDepth, 30);
      return () => clearTimeout(timer);
    }
  }, [open, paintDepth, heightData]);

  // Auto-collapse when no data
  const currentPresetId = reconstruction?.preset_id;
  const depthImageUrl =
    reconstruction?.depth_preview ||
    (currentPresetId ? `/presets/${currentPresetId}/depth.png` : null);

  const hasData = Boolean(opticalUrl || heightData || depthImageUrl);

  const presetLabel = reconstruction?.preset_id
    ? reconstruction.preset_id.toUpperCase()
    : (reconstruction ? 'CUSTOM UPLOAD' : '—');

  return (
    <div className="pointer-events-auto">
      {/* Toggle Button */}
      <button
        id="diagnostic-drawer-toggle"
        onClick={() => setOpen((prev) => !prev)}
        disabled={!hasData}
        className={[
          'flex items-center gap-1.5 px-3 py-1.5 rounded-t-lg font-mono text-[10px] font-bold transition-all backdrop-blur-md shadow-md border border-b-0',
          open
            ? 'bg-cyan-950/90 border-cyan-500/60 text-cyan-300'
            : 'bg-slate-950/85 border-slate-700/60 text-slate-400 hover:text-slate-200',
          !hasData && 'opacity-40 cursor-not-allowed',
        ].join(' ')}
      >
        <FlaskConical className="w-3.5 h-3.5" />
        <span>Pipeline Diagnostics</span>
        {open
          ? <ChevronDown className="w-3 h-3 ml-0.5" />
          : <ChevronUp className="w-3 h-3 ml-0.5" />}
        {/* AOI label */}
        {reconstruction && (
          <span className="ml-1 px-1 py-0.5 rounded text-[8px] bg-slate-800/80 border border-slate-700/60 text-slate-400 uppercase">
            {presetLabel}
          </span>
        )}
      </button>

      {/* Drawer Body */}
      {open && (
        <div className="w-full bg-slate-950/92 border border-cyan-500/30 rounded-lg rounded-tl-none p-3 backdrop-blur-md shadow-hud">

          {/* Header row */}
          <div className="flex items-center justify-between mb-2.5">
            <span className="text-[10px] font-mono font-bold uppercase tracking-widest text-slate-400">
              2D Processing Pipeline — Input → Inference → Sanitization
            </span>
            {reconstruction?.mode && (
              <span className={`text-[9px] font-mono font-bold uppercase px-1.5 py-0.5 rounded border ${
                reconstruction.mode === 'tactical'
                  ? 'bg-emerald-950/80 border-emerald-500/50 text-emerald-300'
                  : 'bg-amber-950/80 border-amber-500/50 text-amber-300'
              }`}>
                {reconstruction.mode === 'tactical' ? 'Calibrated' : 'Relative'}
              </span>
            )}
          </div>

          {/* Three thumbnail cards */}
          <div className="flex gap-2">

            {/* Card 1 — Optical RGB Input */}
            <DiagCard
              id="diag-card-optical"
              title="Input Optical RGB"
              subtitle="Source satellite tile"
              badge="518×518 px"
              badgeColor="cyan"
              lightboxContent={
                opticalUrl
                  ? <img src={opticalUrl} alt="Optical RGB" className="max-w-full max-h-[70vh] object-contain rounded border border-cyan-500/30 shadow-lg" />
                  : <div className="text-slate-500 text-xs font-mono">No optical image</div>
              }
            >
              {opticalUrl ? (
                <img
                  src={opticalUrl}
                  alt="Optical RGB"
                  className="w-full h-full object-cover"
                  draggable={false}
                />
              ) : (
                <div className="w-full h-full flex items-center justify-center">
                  <Image className="w-6 h-6 text-slate-600" />
                </div>
              )}
            </DiagCard>

            {/* Card 2 — Predicted Depth (Inferno Colormap) */}
            <DiagCard
              id="diag-card-depth"
              title="Predicted Depth Map"
              subtitle="Inferno colormap — Depth-Anything-V2"
              badge="256×256 DSM"
              badgeColor="slate"
              lightboxContent={
                <div className="relative w-full h-full flex items-center justify-center p-4 bg-slate-950/80">
                  {depthImageUrl ? (
                    <img 
                      src={depthImageUrl} 
                      alt="Predicted Monocular Depth Map"
                      className="max-w-full max-h-[70vh] rounded border border-cyan-500/30 object-contain shadow-lg"
                      onError={(e) => {
                        console.error("Depth image failed to load, falling back to preset static path:", e);
                        if (currentPresetId) {
                          e.target.src = `/presets/${currentPresetId}/depth.png`;
                        }
                      }}
                    />
                  ) : heightData?.data ? (
                    <canvas
                      ref={depthLightboxCanvasRef}
                      className="max-w-full max-h-[420px] object-contain rounded image-rendering-pixelated"
                      style={{ imageRendering: 'pixelated' }}
                    />
                  ) : (
                    <div className="text-xs font-mono text-cyan-400/60 animate-pulse">
                      Generating or loading depth telemetry...
                    </div>
                  )}
                </div>
              }
            >
              {depthImageUrl ? (
                <img
                  src={depthImageUrl}
                  alt="Predicted Depth"
                  className="w-full h-full object-cover"
                  onError={(e) => {
                    if (currentPresetId) {
                      e.target.src = `/presets/${currentPresetId}/depth.png`;
                    }
                  }}
                />
              ) : heightData?.data ? (
                <canvas
                  ref={depthCanvasRef}
                  className="w-full h-full object-cover"
                  style={{ imageRendering: 'pixelated' }}
                />
              ) : (
                <div className="w-full h-full flex items-center justify-center">
                  <Layers className="w-6 h-6 text-slate-600" />
                </div>
              )}
            </DiagCard>

            {/* Card 3 — Water Sanitization Mask */}
            <DiagCard
              id="diag-card-water"
              title="Water Sanitization Mask"
              subtitle="VWI + Otsu + Luminance Gate"
              badge={waterMaskApplied ? 'Applied' : 'Not Applied'}
              badgeColor={waterMaskApplied ? 'emerald' : 'amber'}
              lightboxContent={
                <div className="flex flex-col items-center gap-4 text-center p-4">
                  {waterMaskApplied ? (
                    <>
                      <CheckCircle2 className="w-12 h-12 text-emerald-400" />
                      <div className="font-mono text-sm text-slate-200 max-w-sm">
                        Shadow-safe water leveling was applied to this reconstruction.
                        Water bodies detected via{' '}
                        <span className="text-cyan-400 font-bold">VWI + Otsu thresholding</span> and
                        confirmed via a{' '}
                        <span className="text-cyan-400 font-bold">luminance gate (τ = 0.25)</span>{' '}
                        to exclude dark mountain shadows.
                      </div>
                      <div className="text-[11px] text-slate-500 font-mono">
                        VWI = (Green − Red) / (Green + Red + 1e−5)
                      </div>
                    </>
                  ) : (
                    <>
                      <AlertTriangle className="w-12 h-12 text-amber-400" />
                      <div className="font-mono text-sm text-slate-200">
                        Water leveling was not applied in this reconstruction.<br/>
                        Enable "Shadow-Safe Water Leveling" and re-run to sanitize water bodies.
                      </div>
                    </>
                  )}
                </div>
              }
            >
              {/* Overlay visual for the mask thumbnail */}
              <div className={`w-full h-full flex flex-col items-center justify-center gap-2 ${waterMaskApplied ? 'bg-emerald-950/40' : 'bg-amber-950/30'}`}>
                <Droplets
                  className={`w-8 h-8 ${waterMaskApplied ? 'text-emerald-400' : 'text-amber-400/60'}`}
                />
                {waterMaskApplied ? (
                  <div className="flex items-center gap-1">
                    <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                    <span className="text-[9px] font-mono text-emerald-300 font-bold">LEVELED</span>
                  </div>
                ) : (
                  <span className="text-[9px] font-mono text-amber-500/80">DISABLED</span>
                )}
              </div>
            </DiagCard>

          </div>

          {/* Footer info row */}
          <div className="mt-2.5 pt-2 border-t border-slate-800/60 flex items-center justify-between text-[9px] font-mono text-slate-500">
            <span>Click any thumbnail to inspect in full resolution</span>
            <span>
              Resolution:{' '}
              <strong className="text-slate-400">
                {reconstruction?.ground_resolution_m_per_px
                  ? `${Number(reconstruction.ground_resolution_m_per_px).toFixed(2)} m/px`
                  : 'Uncalibrated'}
              </strong>
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
