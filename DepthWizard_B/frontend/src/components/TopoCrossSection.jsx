import React, { useMemo } from 'react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from 'recharts';
import { 
  TrendingUp, 
  Ruler, 
  ArrowRight, 
  Mountain, 
  Info,
  Maximize2 
} from 'lucide-react';

/**
 * TopoCrossSection
 * ────────────────────────────────────────────────────────────────────────────
 * Renders the 50-point linear elevation profile curve from the Elevation Ruler.
 * 
 * Displays:
 *   - 2D Cross-section Area Chart (Elevation vs. Distance)
 *   - Cumulative Distance (m / km)
 *   - Altitude Delta (ΔZ, in meters or relative units)
 *   - Grade / Slope % (Maximum & Average along trajectory)
 * ────────────────────────────────────────────────────────────────────────────
 * 
 * @param {Object} props
 * @param {Object|Array} [props.profile] - Profile object or points array:
 *   {
 *     points: Array<{ index: number, distance: number, elevation: number, slope: number }>,
 *     totalDistance?: number,
 *     elevationDelta?: number,
 *     maxSlope?: number,
 *     avgSlope?: number,
 *     minElevation?: number,
 *     maxElevation?: number,
 *     startPoint?: { x: number, y: number },
 *     endPoint?: { x: number, y: number }
 *   }
 * @param {string} [props.units='meters_amsl'] - 'meters_amsl' | 'relative_0_100'
 * @param {string} [props.mode='tactical'] - 'tactical' | 'relative'
 * @param {Function} [props.onHoverPoint] - Optional hover callback
 */
export default function TopoCrossSection({
  profile = null,
  units = 'meters_amsl',
  mode = 'tactical',
  onHoverPoint = null,
}) {
  const isTactical = mode === 'tactical';
  const unitLabel = isTactical ? 'm' : 'rel';

  // Normalize input data
  const { points, summary } = useMemo(() => {
    let rawPoints = [];
    let totalDist = 0;
    let deltaZ = 0;
    let maxSlope = 0;
    let avgSlope = 0;
    let minElev = Infinity;
    let maxElev = -Infinity;

    if (Array.isArray(profile)) {
      rawPoints = profile;
    } else if (profile && Array.isArray(profile.points)) {
      rawPoints = profile.points;
      totalDist = profile.totalDistance ?? 0;
      deltaZ = profile.elevationDelta ?? 0;
      maxSlope = profile.maxSlope ?? 0;
      avgSlope = profile.avgSlope ?? 0;
    }

    if (rawPoints.length > 0) {
      // Format chart-friendly array
      const formatted = rawPoints.map((p, i) => {
        const dist = Number(p.distance ?? i);
        const elev = Number(p.elevation ?? 0);
        const slope = Number(p.slope ?? 0);

        if (elev < minElev) minElev = elev;
        if (elev > maxElev) maxElev = elev;

        return {
          idx: i + 1,
          distance: Math.round(dist * 10) / 10,
          distanceLabel: isTactical 
            ? (dist >= 1000 ? `${(dist / 1000).toFixed(2)} km` : `${Math.round(dist)} m`)
            : `${Math.round(dist)} px`,
          elevation: Math.round(elev * 10) / 10,
          slope: Math.round(slope * 10) / 10,
          point: p,
        };
      });

      // Calculate missing summaries if not provided
      if (!totalDist && formatted.length > 1) {
        totalDist = formatted[formatted.length - 1].distance;
      }
      if (!deltaZ && formatted.length > 1) {
        deltaZ = formatted[formatted.length - 1].elevation - formatted[0].elevation;
      }
      if (!maxSlope && formatted.length > 0) {
        maxSlope = Math.max(...formatted.map(f => Math.abs(f.slope)));
      }
      if (!avgSlope && formatted.length > 0) {
        const sum = formatted.reduce((acc, curr) => acc + Math.abs(curr.slope), 0);
        avgSlope = sum / formatted.length;
      }

      return {
        points: formatted,
        summary: {
          totalDist,
          deltaZ,
          maxSlope,
          avgSlope,
          startElev: formatted[0].elevation,
          endElev: formatted[formatted.length - 1].elevation,
          minElev: minElev === Infinity ? 0 : Math.round(minElev * 10) / 10,
          maxElev: maxElev === -Infinity ? 0 : Math.round(maxElev * 10) / 10,
        },
      };
    }

    return { points: [], summary: null };
  }, [profile, isTactical]);

  // ── INACTIVE EMPTY STATE ───────────────────────────────────────────────────
  if (!points || points.length < 2) {
    return (
      <div className="w-full bg-slate-900/85 border border-slate-800 rounded-lg p-4 text-slate-400 font-mono text-xs backdrop-blur-md flex flex-col items-center justify-center min-h-[190px] text-center">
        <div className="p-2 rounded-full bg-slate-800/80 text-cyan-400 mb-2 border border-slate-700/50">
          <Ruler className="w-5 h-5" />
        </div>
        <span className="font-semibold text-slate-300 uppercase tracking-wider mb-1">
          Elevation Ruler Inactive
        </span>
        <p className="text-[11px] text-slate-500 max-w-sm">
          Click two anchor points on the 3D terrain surface to sample a 50-point linear cross-section profile, distance, and slope gradient.
        </p>
      </div>
    );
  }

  // Calculate Y-axis domain padding
  const yDomain = [
    Math.floor(summary.minElev * 0.98),
    Math.ceil(summary.maxElev * 1.02),
  ];

  return (
    <div className="w-full bg-slate-900/90 border border-slate-800 rounded-lg p-3 sm:p-4 text-slate-200 backdrop-blur-md shadow-hud font-sans">
      {/* Header bar */}
      <div className="flex flex-wrap items-center justify-between gap-2 pb-2.5 mb-2.5 border-b border-slate-800/80">
        <div className="flex items-center gap-2 font-mono">
          <TrendingUp className="w-4 h-4 text-cyan-400" />
          <span className="text-xs font-bold uppercase tracking-wider text-slate-100">
            Topographic Cross-Section Profile
          </span>
          <span className="px-1.5 py-0.5 rounded text-[10px] bg-cyan-950/80 border border-cyan-500/40 text-cyan-300">
            50 Samples
          </span>
        </div>

        <div className="text-[11px] font-mono text-slate-400 flex items-center gap-3">
          <span className="flex items-center gap-1">
            <span className="w-2 h-2 rounded-full bg-cyan-400 inline-block"></span>
            Elevation ({unitLabel})
          </span>
          <span className="text-slate-600">|</span>
          <span>
            Mode: <strong className={isTactical ? 'text-emerald-400' : 'text-amber-400'}>{mode.toUpperCase()}</strong>
          </span>
        </div>
      </div>

      {/* Chart container */}
      <div className="w-full h-44 sm:h-48 font-mono text-[10px]">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart
            data={points}
            margin={{ top: 10, right: 10, left: -15, bottom: 0 }}
            onMouseMove={(state) => {
              if (onHoverPoint && state && state.activePayload && state.activePayload.length > 0) {
                onHoverPoint(state.activePayload[0].payload);
              }
            }}
            onMouseLeave={() => {
              if (onHoverPoint) onHoverPoint(null);
            }}
          >
            <defs>
              <linearGradient id="topoElevationGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.4} />
                <stop offset="95%" stopColor="#06b6d4" stopOpacity={0.0} />
              </linearGradient>
            </defs>

            <CartesianGrid strokeDasharray="2 2" stroke="#1e293b" vertical={false} />

            <XAxis
              dataKey="distance"
              stroke="#64748b"
              tickLine={false}
              tickFormatter={(v) => (isTactical ? `${Math.round(v)}m` : `${Math.round(v)}px`)}
              fontSize={10}
            />

            <YAxis
              domain={yDomain}
              stroke="#64748b"
              tickLine={false}
              tickFormatter={(v) => `${Math.round(v)}`}
              fontSize={10}
            />

            <Tooltip
              content={({ active, payload }) => {
                if (!active || !payload || !payload.length) return null;
                const d = payload[0].payload;
                return (
                  <div className="bg-slate-950/95 border border-cyan-500/50 p-2.5 rounded shadow-hud-glow font-mono text-[11px] text-slate-200">
                    <div className="text-cyan-400 font-bold border-b border-slate-800 pb-1 mb-1.5 flex justify-between gap-4">
                      <span>SAMPLE #{d.idx} / 50</span>
                      <span>{d.distanceLabel}</span>
                    </div>
                    <div className="grid grid-cols-2 gap-x-3 gap-y-0.5">
                      <span className="text-slate-400">Elevation:</span>
                      <span className="text-right font-semibold text-white">
                        {d.elevation} {unitLabel}
                      </span>
                      <span className="text-slate-400">Slope:</span>
                      <span className={`text-right font-semibold ${Math.abs(d.slope) > 20 ? 'text-rose-400' : 'text-emerald-400'}`}>
                        {d.slope > 0 ? `+${d.slope}` : d.slope}%
                      </span>
                    </div>
                  </div>
                );
              }}
            />

            <Area
              type="monotone"
              dataKey="elevation"
              stroke="#06b6d4"
              strokeWidth={2}
              fill="url(#topoElevationGrad)"
              isAnimationActive={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Summary Telemetry Metrics Footer */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 mt-3 pt-3 border-t border-slate-800/80 font-mono text-xs">
        {/* Distance */}
        <div className="bg-slate-950/60 border border-slate-800/80 rounded p-2">
          <span className="text-[10px] uppercase text-slate-400 block mb-0.5">Distance</span>
          <div className="text-slate-100 font-bold">
            {isTactical ? (
              summary.totalDist >= 1000 
                ? `${(summary.totalDist / 1000).toFixed(2)} km` 
                : `${summary.totalDist.toFixed(1)} m`
            ) : (
              `${summary.totalDist.toFixed(0)} px`
            )}
          </div>
          <span className="text-[10px] text-slate-500">Horizontal path</span>
        </div>

        {/* Altitude Delta */}
        <div className="bg-slate-950/60 border border-slate-800/80 rounded p-2">
          <span className="text-[10px] uppercase text-slate-400 block mb-0.5">Altitude Delta (ΔZ)</span>
          <div className="font-bold flex items-center gap-1">
            <span className={summary.deltaZ >= 0 ? 'text-emerald-400' : 'text-rose-400'}>
              {summary.deltaZ >= 0 ? `+${summary.deltaZ.toFixed(1)}` : summary.deltaZ.toFixed(1)}
            </span>
            <span className="text-[10px] text-slate-400 font-sans">{unitLabel}</span>
          </div>
          <span className="text-[10px] text-slate-500">
            {summary.startElev.toFixed(0)} → {summary.endElev.toFixed(0)} {unitLabel}
          </span>
        </div>

        {/* Max Slope */}
        <div className="bg-slate-950/60 border border-slate-800/80 rounded p-2">
          <span className="text-[10px] uppercase text-slate-400 block mb-0.5">Max Slope</span>
          <div className={`font-bold ${summary.maxSlope > 30 ? 'text-rose-400' : 'text-amber-400'}`}>
            {summary.maxSlope.toFixed(1)}%
          </div>
          <span className="text-[10px] text-slate-500">Peak grade</span>
        </div>

        {/* Avg Slope */}
        <div className="bg-slate-950/60 border border-slate-800/80 rounded p-2">
          <span className="text-[10px] uppercase text-slate-400 block mb-0.5">Avg Slope</span>
          <div className="text-slate-100 font-bold">
            {summary.avgSlope.toFixed(1)}%
          </div>
          <span className="text-[10px] text-slate-500">Mean gradient</span>
        </div>
      </div>
    </div>
  );
}
