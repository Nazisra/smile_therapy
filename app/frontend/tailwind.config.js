/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'BlinkMacSystemFont', '"Segoe UI"', 'Roboto', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
      colors: {
        burgundy: {
          50: '#fdf4f7',
          100: '#fbe8f0',
          200: '#f7d3e2',
          300: '#f0b0cb',
          400: '#e582ac',
          500: '#d5588c',
          600: '#bd396f',
          700: '#9f2858',
          800: '#832349',
          900: '#6f2140',
          950: '#270e17',
        }
      }
    },
  },
  plugins: [],
}
