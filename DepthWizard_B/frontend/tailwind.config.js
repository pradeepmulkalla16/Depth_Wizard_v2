/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        isro: {
          orange: '#FF9933',
          navy: '#060d1a',
          space: '#030712',
          cyan: '#00e5ff',
        },
        tactical: {
          950: '#030712',
          900: '#0b0f19',
          850: '#111827',
          800: '#1e293b',
          700: '#334155',
          cyan: '#06b6d4',
          emerald: '#10b981',
          amber: '#f59e0b',
          rose: '#f43f5e',
          accent: '#38bdf8',
        },
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'Monaco', 'Consolas', 'monospace'],
        sans: ['Inter', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
      },
      boxShadow: {
        'hud': '0 0 25px -5px rgba(6, 182, 212, 0.2), inset 0 1px 1px rgba(255, 255, 255, 0.05)',
        'hud-glow': '0 0 15px rgba(56, 189, 248, 0.35)',
        'radar': '0 0 40px rgba(0, 229, 255, 0.15)',
      },
      animation: {
        'pulse-subtle': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'radar-sweep': 'spin 4s linear infinite',
      },
      backdropBlur: {
        'xs': '2px',
      },
    },
  },
  plugins: [],
}
