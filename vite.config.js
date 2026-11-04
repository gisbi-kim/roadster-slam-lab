import { defineConfig } from 'vite';
export default defineConfig({ base: '/roadster-slam-lab/', build: { rollupOptions: { input: ['index.html', 'prior.html'] } } });
