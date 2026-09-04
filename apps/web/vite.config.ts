import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
//
// Pinned to classic esbuild-based Vite 5 (not the Vite 8 / rolldown-vite
// default from `npm create vite@latest`). rolldown-vite's dependency
// optimizer mishandles maplibre-gl's internal Web Worker (dynamic URL
// import) - it kept re-pre-bundling maplibre-gl into .vite/deps/ even with
// optimizeDeps.exclude set, and the browser threw "does not provide an
// export named 'default'" no matter what cache-busting was tried. Vite 5's
// esbuild optimizer is the well-established, documented-working combo with
// maplibre-gl (it's what maplibre-gl's own examples and most production
// React+MapLibre apps use), so no exclude/workaround should be needed here.
export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      output: {
        // maplibre-gl alone is ~800KB unminified - splitting it into its
        // own chunk means a repeat visit (or a future page that doesn't
        // need the map) can skip re-downloading it once it's cached,
        // rather than it being fused into one 1.16MB bundle with app code
        // that changes far more often.
        manualChunks: { maplibre: ["maplibre-gl"] },
      },
    },
  },
})
