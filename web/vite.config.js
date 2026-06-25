import { defineConfig } from 'vite';

// Static SPA. public/ (_headers) is copied verbatim to dist/.
export default defineConfig({
  build: { target: 'es2022' },
});
