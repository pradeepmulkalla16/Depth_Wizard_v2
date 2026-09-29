import React, { useState } from 'react';
import { 
  ShieldCheck, 
  AlertTriangle, 
  Activity, 
  Layers, 
  ArrowUpDown, 
  Cpu, 
  Compass, 
  CheckCircle2, 
  Radio, 
  Satellite, 
  MapPin, 
  X,
  ChevronRight,
} from 'lucide-react';

/**
 * ValidationHUD
 * ────────────────────────────────────────────────────────────────────────────
 * Collapsible right-docked operational status badge & validation metrics HUD.
 *
 * COLLAPSED STATE (Default):
 *   Compact glassmorphic floating pill anchored at top-right of the 3D viewport.
 *   - Mode A (Calibrated): "🟢 CALIBRATED TACTICAL (METRIC AMSL) ◂" (Emerald)
 *   - Mode B (Relative):   "🟡 RAPID RECONNAISSANCE ◂" (Amber)
 *   Clicking the pill slides open the full validation panel.
 *
 * EXPANDED STATE:
 *   Sleek glassmorphic card from the right edge with a close button (✕ / ▸).
 *   Reveals:
 *   - Mode badge & Target AOI
 *   - Affine fit equation strip (Z = s·d̂ + t)
 *   - Scale (s), Datum shift (t)
 *   - Pearson r, MAE, RMSE, GSD, and Inversion status.
 * ────────────────────────────────────────────────────────────────────────────
 *
 * @param {string}      mode             - 'tactical' | 'relative'
 * @param {Object|null} calibration      - { s, t, depth_inverted, fit_method }
 * @param {Object|null} metrics          - { r, mae, rmse }
 * @param {number|null} groundResolution - ground resolution in m/px
 * @param {string|null} presetId         - active preset region ID
 */
export default function ValidationHUD({
  mode = 'relative',
  calibration = null,
  metrics = null,
  groundResolution = null,
  presetId = null,
}) {
  const [isOpen, setIsOpen] = useState(false);
  const isTactical = mode === 'tactical' && Boolean(calibration);

  // ── COLLAPSED STATE (DEFAULT) ─────────────────────────────────────────────
  if (!isOpen) {
    if (isTactical) {
      return (
        <button
          onClick={() => setIsOpen(true)}
          title="Click to view full calibration telemetry"
          aria-label="Open Calibrated Tactical Validation HUD"
          className="pointer-events-auto group px-3.5 py-2 rounded-full bg-slate-950/85 hover:bg-slate-900/95 border border-emerald-500/50 hover:border-emerald-400 text-emerald-300 font-mono text-xs font-bold tracking-wide shadow-[0_0_15px_rgba(16,185,129,0.25)] hover:shadow-[0_0_20px_rgba(16,185,129,0.45)] backdrop-blur-md flex items-center gap-2 cursor-pointer transition-all duration-200 ease-in-out select-none active:scale-95"
        >
          <span className="flex h-2 w-2 relative">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
          </span>
          <span>🟢 CALIBRATED TACTICAL (METRIC AMSL)</span>
          <span className="text-emerald-400 font-bold transition-transform group-hover:-translate-x-0.5">◂</span>
        </button>
      );
    }

    return (
      <button
        onClick={() => setIsOpen(true)}
        title="Click to view reconnaissance status"
        aria-label="Open Rapid Reconnaissance Status HUD"
        className="pointer-events-auto group px-3.5 py-2 rounded-full bg-slate-950/85 hover:bg-slate-900/95 border border-amber-500/50 hover:border-amber-400 text-amber-300 font-mono text-xs font-bold tracking-wide shadow-[0_0_15px_rgba(245,158,11,0.25)] hover:shadow-[0_0_20px_rgba(245,158,11,0.45)] backdrop-blur-md flex items-center gap-2 cursor-pointer transition-all duration-200 ease-in-out select-none active:scale-95"
      >
        <span className="flex h-2 w-2 relative">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-amber-400 opacity-75"></span>
          <span className="relative inline-flex rounded-full h-2 w-2 bg-amber-500"></span>
        </span>
        <span>🟡 RAPID RECONNAISSANCE</span>
        <span className="text-amber-400 font-bold transition-transform group-hover:-translate-x-0.5">◂</span>
      </button>
    );
  }

  // ── EXPANDED STATE ────────────────────────────────────────────────────────

  // MODE B — RAPID RECONNAISSANCE
  if (!isTactical) {
    return (
      <div className="pointer-events-auto w-full max-w-sm sm:max-w-md bg-slate-950/90 border border-amber-500/50 rounded-xl p-3.5 sm:p-4 text-slate-200 backdrop-blur-md shadow-hud font-sans transition-all duration-200 ease-in-out">
        {/* Header with Close Button */}
        <div className="flex items-center justify-between gap-2 pb-3 mb-3 border-b border-slate-800/80">
          <div className="flex items-center gap-2">
            <div className="p-1.5 rounded-lg bg-amber-500/15 border border-amber-500/40">
              <Radio className="w-4 h-4 text-amber-400 animate-pulse" />
            </div>
            <div>
              <div className="flex items-center gap-2 flex-wrap">
                <span className="text-xs font-bold tracking-widest uppercase text-amber-400 font-mono">
                  🟡 Rapid Reconnaissance
                </span>
                <span className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-amber-950/80 border border-amber-500/40 text-amber-300 font-bold uppercase tracking-wide">
                  Relative rDSM
                </span>
              </div>
              <p className="text-[11px] text-slate-400 mt-0.5 font-mono">
                Uncalibrated optical input — no GCPs available
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {presetId && (
              <div className="text-right mr-1">
                <span className="text-[10px] uppercase tracking-wider text-slate-500 block font-mono">AOI</span>
                <span className="text-xs font-mono font-semibold text-slate-300 uppercase">{presetId}</span>
              </div>
            )}
            <button
              onClick={() => setIsOpen(false)}
              title="Collapse Panel (▸)"
              aria-label="Collapse Validation HUD"
              className="p-1.5 rounded-lg bg-slate-900 border border-slate-800 hover:border-amber-500/50 hover:bg-slate-800 text-slate-400 hover:text-amber-300 transition-all flex items-center justify-center cursor-pointer"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Honest scale information */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 font-mono text-xs">
          <div className="bg-slate-950/70 border border-amber-500/20 rounded p-2.5">
            <div className="flex items-center justify-between text-slate-400 text-[10px] mb-1">
              <span className="uppercase tracking-wider">Vertical Scale</span>
              <Layers className="w-3 h-3 text-amber-400" />
            </div>
            <div className="text-sm font-bold text-amber-300">Dimensionless</div>
            <span className="text-[10px] text-slate-500">Relative Units [0.0 – 1.0]</span>
          </div>

          <div className="bg-slate-950/70 border border-amber-500/20 rounded p-2.5">
            <div className="flex items-center justify-between text-slate-400 text-[10px] mb-1">
              <span className="uppercase tracking-wider">GCP Status</span>
              <MapPin className="w-3 h-3 text-amber-400" />
            </div>
            <div className="text-sm font-bold text-amber-300">Uncalibrated</div>
            <span className="text-[10px] text-slate-500">GCPs required for metric AMSL</span>
          </div>
        </div>

        {/* Honest disclaimer */}
        <div className="mt-2.5 pt-2 border-t border-slate-800/70 flex items-start gap-1.5 text-[10px] text-slate-500 font-mono">
          <AlertTriangle className="w-3 h-3 text-amber-500/70 flex-shrink-0 mt-0.5" />
          <span>
            Relative reconstruction only. Upload image provides plausible 3D relief geometry
            without real-world elevation values. Use a calibrated preset for metric AMSL output.
          </span>
        </div>
      </div>
    );
  }

  // MODE A — CALIBRATED TACTICAL
  const { r, mae, rmse } = metrics || {};
  const { s, t, depth_inverted, fit_method } = calibration || {};
  const isRStrong = typeof r === 'number' && r >= 0.85;

  return (
    <div className="pointer-events-auto w-full max-w-sm sm:max-w-md lg:max-w-lg bg-slate-950/90 border border-emerald-500/50 rounded-xl p-3.5 sm:p-4 text-slate-200 backdrop-blur-md shadow-hud font-sans transition-all duration-200 ease-in-out">
      {/* Top Header with Close Button */}
      <div className="flex items-center justify-between gap-2 pb-3 mb-3 border-b border-slate-800/80">
        <div className="flex items-center gap-2">
          <div className="p-1.5 rounded-lg bg-emerald-500/15 border border-emerald-500/40">
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
          </div>
          <div>
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-xs font-bold tracking-widest uppercase text-emerald-400 font-mono">
                🟢 Calibrated Tactical
              </span>
              <span className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-emerald-950/80 border border-emerald-500/40 text-emerald-300 font-bold uppercase tracking-wide">
                Metric AMSL
              </span>
            </div>
            <p className="text-[11px] text-slate-400 mt-0.5 font-mono">
              Validated against holdout GCP ground truth (independent split)
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {presetId && (
            <div className="text-right mr-1">
              <span className="text-[10px] uppercase tracking-wider text-slate-500 block font-mono">
                AOI
              </span>
              <span className="text-xs font-mono font-semibold text-slate-300 uppercase">
                {presetId}
              </span>
            </div>
          )}
          <button
            onClick={() => setIsOpen(false)}
            title="Collapse Panel (▸)"
            aria-label="Collapse Validation HUD"
            className="p-1.5 rounded-lg bg-slate-900 border border-slate-800 hover:border-emerald-500/50 hover:bg-slate-800 text-slate-400 hover:text-emerald-300 transition-all flex items-center justify-center cursor-pointer"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Affine Calibration Parameters Strip: Z = s·d̂ + t */}
      <div className="flex flex-wrap items-center gap-2 sm:gap-3 mb-3 px-2.5 py-1.5 rounded bg-slate-900/80 border border-slate-800/80 font-mono text-[10px] text-slate-400">
        <Satellite className="w-3 h-3 text-cyan-400 flex-shrink-0" />
        <span>
          <span className="text-slate-500">Affine Fit:</span>{' '}
          <span className="text-slate-200">Z = </span>
          <span className="text-cyan-400 font-bold">s</span>
          <span className="text-slate-200">·d̂ + </span>
          <span className="text-sky-400 font-bold">t</span>
        </span>
        <span className="text-slate-600">|</span>
        <span>
          <span className="text-slate-500">Datum shift </span>
          <span className="text-sky-400 font-bold">t: </span>
          <span className="text-slate-200">{typeof t === 'number' ? `${t >= 0 ? '+' : ''}${t.toFixed(1)} m` : '—'}</span>
        </span>
        <span className="text-slate-600 hidden sm:inline">|</span>
        <span className="hidden sm:inline">
          <span className="text-slate-500">Scale </span>
          <span className="text-cyan-400 font-bold">s: </span>
          <span className="text-slate-200">{typeof s === 'number' ? `${s.toFixed(3)} m/unit` : '—'}</span>
        </span>
      </div>

      {/* Full Metrics Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-2 font-mono">
        {/* Pearson r */}
        <div className="bg-slate-950/70 border border-slate-800/90 rounded p-2 relative overflow-hidden">
          <div className="flex items-center justify-between text-slate-400 text-[10px] mb-1">
            <span className="uppercase tracking-wider">Pearson r</span>
            <Activity className="w-3 h-3 text-cyan-400" />
          </div>
          <div className="flex items-baseline gap-1">
            <span className="text-xs text-slate-500">r =</span>
            <span className={`text-base font-bold ${isRStrong ? 'text-emerald-400' : 'text-amber-400'}`}>
              {typeof r === 'number' ? r.toFixed(3) : '—'}
            </span>
          </div>
          <span className="text-[9px] text-slate-500 block truncate">
            {isRStrong ? 'Strong alignment' : 'Moderate correlation'}
          </span>
        </div>

        {/* MAE */}
        <div className="bg-slate-950/70 border border-slate-800/90 rounded p-2">
          <div className="flex items-center justify-between text-slate-400 text-[10px] mb-1">
            <span className="uppercase tracking-wider">MAE</span>
            <Compass className="w-3 h-3 text-cyan-400" />
          </div>
          <div className="flex items-baseline gap-1">
            <span className="text-base font-bold text-slate-100">
              {typeof mae === 'number' ? `±${mae.toFixed(2)}` : '—'}
            </span>
            <span className="text-xs text-slate-400 font-sans">m</span>
          </div>
          <span className="text-[9px] text-slate-500 block truncate">Holdout residual</span>
        </div>

        {/* RMSE */}
        <div className="bg-slate-950/70 border border-slate-800/90 rounded p-2">
          <div className="flex items-center justify-between text-slate-400 text-[10px] mb-1">
            <span className="uppercase tracking-wider">RMSE</span>
            <Layers className="w-3 h-3 text-cyan-400" />
          </div>
          <div className="flex items-baseline gap-1">
            <span className="text-base font-bold text-slate-100">
              {typeof rmse === 'number' ? rmse.toFixed(2) : '—'}
            </span>
            <span className="text-xs text-slate-400 font-sans">m</span>
          </div>
          <span className="text-[9px] text-slate-500 block truncate">Precision spread</span>
        </div>

        {/* Scale & Datum Offset */}
        <div className="bg-slate-950/70 border border-slate-800/90 rounded p-2">
          <div className="flex items-center justify-between text-slate-400 text-[10px] mb-1">
            <span className="uppercase tracking-wider">Scale / Shift</span>
            <Cpu className="w-3 h-3 text-cyan-400" />
          </div>
          <div className="text-xs font-semibold text-slate-200 truncate">
            s: <span className="text-cyan-400">{typeof s === 'number' ? s.toFixed(3) : '—'}</span>
          </div>
          <div className="text-[10px] text-slate-400 truncate">
            t: <span className="text-slate-300">{typeof t === 'number' ? `${t.toFixed(1)}m` : '—'}</span>
          </div>
        </div>

        {/* Inversion & Fit Method */}
        <div className="col-span-2 sm:col-span-1 bg-slate-950/70 border border-slate-800/90 rounded p-2">
          <div className="flex items-center justify-between text-slate-400 text-[10px] mb-1">
            <span className="uppercase tracking-wider">Inversion</span>
            <ArrowUpDown className="w-3 h-3 text-cyan-400" />
          </div>
          <div className="flex items-center gap-1">
            {depth_inverted ? (
              <span className="text-xs font-bold text-amber-400">FLIPPED</span>
            ) : (
              <span className="text-xs font-bold text-emerald-400">NORMAL</span>
            )}
          </div>
          <span className="text-[9px] text-slate-500 truncate block">
            {fit_method || 'RANSAC'}
          </span>
        </div>
      </div>

      {/* Footer Sub-bar with GSD */}
      {groundResolution && (
        <div className="mt-2.5 pt-2 border-t border-slate-800/70 flex items-center justify-between text-[11px] text-slate-400 font-mono">
          <span className="flex items-center gap-1">
            <CheckCircle2 className="w-3 h-3 text-emerald-400" />
            True metric AMSL elevation
          </span>
          <span>
            GSD: <strong className="text-slate-200">{Number(groundResolution).toFixed(2)} m/px</strong>
          </span>
        </div>
      )}
    </div>
  );
}
