/// <reference types="vitest" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/health": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/btc-price": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/simulate": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/optimize": { target: "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
  test: {
    environment: "happy-dom",
    include: ["src/**/*.test.{ts,tsx}"],
  },
});
