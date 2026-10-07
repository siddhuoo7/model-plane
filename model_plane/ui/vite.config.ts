import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // Base URL matches the FastAPI static mount at /admin
  base: '/admin/',
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
  // Suppress Sass deprecation warnings from @carbon/styles internals.
  // These originate in node_modules (if-function syntax), not in our code.
  css: {
    preprocessorOptions: {
      scss: {
        silenceDeprecations: ['if-function'],
      },
    },
  },
  server: {
    proxy: {
      // In dev, proxy /admin/api/* and /v1/* to the FastAPI backend.
      // changeOrigin is required for streaming responses (SSE, chat completions).
      '/admin/api': {
        target: 'http://localhost:8081',
        changeOrigin: true,
      },
      '/v1': {
        target: 'http://localhost:8081',
        changeOrigin: true,
      },
    },
  },
})
