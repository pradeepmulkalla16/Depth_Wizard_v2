import React from 'react';
import { Layers, Thermometer, Grid3x3 } from 'lucide-react';

/**
 * ViewportShaderToggle
 * ────────────────────────────────────────────────────────────────────────────
 * Glassmorphic floating 3-segment pill docked at top-center of the 3D viewport.
 * Renders as a plain DOM overlay (outside the R3F <Canvas>) so it never triggers
 * a Three.js unmount or disrupts OrbitControls / flythrough state.
 *
 * Synchronised with the sidebar shader buttons via shared `colorMode` state
 * lifted in App.jsx — both controls always reflect the same active mode.
 *
 * @param {'optical'|'hypsometric'|'wireframe'} colorMode - Current active shader mode
 * @param {(mode: string) => void} onModeChange - Callback to update parent state
 */
export default function ViewportShaderToggle({ colorMode = 'optical', onModeChange }) {
  const MODES = [
    {
      id: 'optical',
      label: 'RGB Texture',
      shortLabel: 'RGB',
      Icon: Layers,
      activeClass: 'bg-sky-500/90 text-slate-950 shadow-[0_0_12px_rgba(14,165,233,0.5)]',
      iconColor: 'text-slate-950',
    },
    {
      id: 'hypsometric',
      label: 'Elev. Heatmap',
      shortLabel: 'Heatmap',
      Icon: Thermometer,
      activeClass: 'bg-emerald-500/90 text-slate-950 shadow-[0_0_12px_rgba(16,185,129,0.5)]',
      iconColor: 'text-slate-950',
    },
    {
      id: 'wireframe',
      label: 'Wireframe',
      shortLabel: 'Wire',
      Icon: Grid3x3,
      activeClass: 'bg-cyan-400/90 text-slate-950 shadow-[0_0_12px_rgba(34,211,238,0.5)]',
      iconColor: 'text-slate-950',
    },
  ];

  return (
    <div
      className="absolute top-3 left-1/2 -translate-x-1/2 z-20 pointer-events-auto"
      role="group"
      aria-label="Surface shader mode"
    >
      <div className="flex items-center gap-0 p-0.5 rounded-full bg-slate-950/80 border border-slate-700/60 backdrop-blur-md shadow-hud">
        {MODES.map((mode, idx) => {
          const isActive = colorMode === mode.id;
          const { Icon } = mode;
          return (
            <button
              key={mode.id}
              id={`shader-toggle-${mode.id}`}
              onClick={() => onModeChange?.(mode.id)}
              title={mode.label}
              className={[
                'flex items-center gap-1.5 px-3 py-1.5 rounded-full font-mono text-[11px] font-bold transition-all duration-200 select-none',
                isActive
                  ? mode.activeClass
                  : 'text-slate-400 hover:text-white hover:bg-slate-800/60',
              ].join(' ')}
            >
              <Icon className={`w-3 h-3 flex-shrink-0 ${isActive ? mode.iconColor : ''}`} />
              {/* Full label on wider screens, abbreviated on small */}
              <span className="hidden sm:inline">{mode.label}</span>
              <span className="sm:hidden">{mode.shortLabel}</span>
            </button>
          );
        })}
      </div>

      {/* Subtle subtitle showing active mode */}
      <div className="text-center mt-1 pointer-events-none">
        <span className="text-[9px] font-mono text-slate-500 uppercase tracking-widest">
          Surface Shader
        </span>
      </div>
    </div>
  );
}
