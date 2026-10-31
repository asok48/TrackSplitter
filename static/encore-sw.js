// Encore's offline copy (static/encore-manifest.json makes Encore an installable Chrome app): keeps the
// game's own files so the app window can open even when Track Splitter isn't running, show "Starting
// Track Splitter…" and start it through the helper (see index.html, startTrackSplitter). Always tries the
// server first, so updates show up as soon as it's running; the API is never cached.
const CACHE = 'encore-v1';
const FILES = ['/game', '/game3d.js', '/encore-manifest.json', '/icons/encore-192.png', '/icons/encore-512.png',
  '/vendor/three/build/three.module.js',
  '/vendor/three/examples/jsm/geometries/RoundedBoxGeometry.js', '/vendor/three/examples/jsm/environments/RoomEnvironment.js',
  '/vendor/three/examples/jsm/postprocessing/EffectComposer.js', '/vendor/three/examples/jsm/postprocessing/RenderPass.js',
  '/vendor/three/examples/jsm/postprocessing/UnrealBloomPass.js', '/vendor/three/examples/jsm/postprocessing/OutputPass.js',
  '/vendor/three/examples/jsm/postprocessing/Pass.js', '/vendor/three/examples/jsm/postprocessing/ShaderPass.js',
  '/vendor/three/examples/jsm/postprocessing/MaskPass.js', '/vendor/three/examples/jsm/shaders/CopyShader.js',
  '/vendor/three/examples/jsm/shaders/LuminosityHighPassShader.js', '/vendor/three/examples/jsm/shaders/OutputShader.js'];

self.addEventListener('install', (e) => { e.waitUntil(caches.open(CACHE).then((c) => c.addAll(FILES)).then(() => self.skipWaiting())); });
self.addEventListener('activate', (e) => { e.waitUntil(self.clients.claim()); });
self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== 'GET' || url.origin !== location.origin || url.pathname.startsWith('/api/')) return;
  const key = url.pathname === '/game' || url.pathname === '/game/' ? '/game' : url.pathname;
  if (!FILES.includes(key)) return;
  e.respondWith(fetch(e.request).then((r) => {
    if (r.ok) { const copy = r.clone(); caches.open(CACHE).then((c) => c.put(key, copy)); }
    return r;
  }).catch(() => caches.match(key)));
});
