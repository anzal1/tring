import path from "node:path";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Builds straight into the Python package: the wheel ships the compiled app,
// so `pip install tring` users never need Node. `base: "./"` keeps asset URLs
// relative, letting the studio server mount anywhere.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: "./",
  resolve: {
    alias: { "@": path.resolve(__dirname, "./src") },
  },
  build: {
    outDir: "../src/tring/studio/static",
    emptyOutDir: true,
  },
  server: {
    // dev loop: `npm run dev` proxies API/WS to a running studio server
    proxy: {
      "/api": "http://127.0.0.1:8977",
      "/ws": { target: "ws://127.0.0.1:8977", ws: true },
    },
  },
});
