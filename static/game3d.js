// The game's 3D board (user: the 2D board "looks very POC" / "flash gamey"; prototype in
// design/game-mockups/board-3d-prototype.html). A rendered scene with three.js (static/vendor/three,
// MIT, so it works offline): the board lies sideways like a tab (notes travel right to left, low E
// at the bottom; a vertical track was found too confusing) and leans back 45 degrees; a maple or dark
// fretboard with fret wires as bar lines; solid glossy notes, all one size, kept apart by the board's
// speed (quick repeats of the same note are slim ticks); low pads at the strike line that dip and
// light up on a hit; warm stage lights that follow how you're doing, blue at x4.
// index.html drives it each frame with Game3D.frame(view) and draws the HUD on a canvas over it.
import * as THREE from 'three';
import { RoundedBoxGeometry } from '/vendor/three/examples/jsm/geometries/RoundedBoxGeometry.js';
import { RoomEnvironment } from '/vendor/three/examples/jsm/environments/RoomEnvironment.js';
import { EffectComposer } from '/vendor/three/examples/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from '/vendor/three/examples/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from '/vendor/three/examples/jsm/postprocessing/UnrealBloomPass.js';
import { OutputPass } from '/vendor/three/examples/jsm/postprocessing/OutputPass.js';

const LANE = 1.5, NUT = -0.62, NOTE_W = 0.56, GAP = 0.16, VISIBLE = 17.5, FADE = 3.5;
const FONT = '"Avenir Next Condensed", Impact, sans-serif';

let renderer, composer, scene, camera, world, board, mats, wrapEl = null;
let lanes = { n: 0 }, bars = [], pool = [], used = new Map(), sparks = [], seenBurst = 0, lastSize = '';
const tex = {};
const v3 = new THREE.Vector3();

function label(text, color, outline = false, size = 256, chip = false) {   // 256 px: crisp up close ("higher res")
  const c = document.createElement('canvas'); c.width = c.height = size; const g = c.getContext('2d');
  if (chip) {                                            // a small glass chip behind (the string names)
    g.fillStyle = 'rgba(10,8,6,0.6)'; g.strokeStyle = 'rgba(255,245,230,0.22)'; g.lineWidth = size * 0.03;
    g.beginPath(); g.roundRect(size * 0.14, size * 0.14, size * 0.72, size * 0.72, size * 0.2); g.fill(); g.stroke();
    outline = false;
  }
  g.font = `900 ${size * (chip ? 0.5 : 0.74)}px ${FONT}`; g.textAlign = 'center'; g.textBaseline = 'middle';
  if (outline) { g.lineWidth = size * 0.1; g.lineJoin = 'round'; g.strokeStyle = 'rgba(10,7,5,.9)'; g.strokeText(text, size / 2, size / 2 + size * 0.05); }
  g.fillStyle = color; g.fillText(text, size / 2, size / 2 + size * 0.05);
  const t = new THREE.CanvasTexture(c); t.anisotropy = 8; t.colorSpace = THREE.SRGBColorSpace; return t;
}

// Wood: long, continuous, gently wavy grain along the neck, a few broad soft streaks; whole numbers of
// waves across the texture so it repeats without a seam (scattered streaks looked "sloppy").
function woodTexture(base, light, dark, contrast) {
  const W = 2048, H = 256, c = document.createElement('canvas'); c.width = W; c.height = H; const g = c.getContext('2d');
  g.fillStyle = base; g.fillRect(0, 0, W, H);
  for (let i = 0; i < 6; i++) {
    const y = Math.random() * H, h = 20 + Math.random() * 50, grad = g.createLinearGradient(0, y - h, 0, y + h);
    grad.addColorStop(0, 'rgba(0,0,0,0)'); grad.addColorStop(0.5, Math.random() < 0.5 ? light : dark); grad.addColorStop(1, 'rgba(0,0,0,0)');
    g.globalAlpha = 0.18 * contrast; g.fillStyle = grad; g.fillRect(0, y - h, W, h * 2);
  }
  g.lineWidth = 1.2;
  for (let i = 0; i < 70; i++) {
    const y0 = Math.random() * H, amp = 2 + Math.random() * 6, k1 = 1 + Math.floor(Math.random() * 3), k2 = 4 + Math.floor(Math.random() * 5), ph = Math.random() * 6;
    g.globalAlpha = (0.12 + Math.random() * 0.22) * contrast; g.strokeStyle = Math.random() < 0.8 ? dark : light;
    g.beginPath();
    for (let x = 0; x <= W; x += 16) { const a = (2 * Math.PI * x) / W, y = y0 + Math.sin(k1 * a + ph) * amp + Math.sin(k2 * a + ph * 2) * amp * 0.25; if (x) g.lineTo(x, y); else g.moveTo(x, y); }
    g.stroke();
  }
  g.globalAlpha = 1;
  const t = new THREE.CanvasTexture(c); t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(6, 1); t.anisotropy = 8; t.colorSpace = THREE.SRGBColorSpace; return t;
}

function init() {
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });   // see-through: the song's blurred cover shows behind (index.html)
  renderer.setPixelRatio(Math.min(3, devicePixelRatio));   // the screen's full sharpness
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.domElement.className = 'game3d-gl';
  scene = new THREE.Scene();
  scene.background = null;
  scene.fog = new THREE.Fog('#0b0806', 15, 36);
  scene.environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.04).texture;
  scene.environmentIntensity = 0.3;
  camera = new THREE.PerspectiveCamera(40, 16 / 10, 0.1, 100);
  world = new THREE.Group(); world.rotation.x = Math.PI / 4; world.scale.z = 1.2; scene.add(world);   // leaning back 45 degrees

  scene.add(new THREE.HemisphereLight('#ffd7a8', '#0b0806', 0.25));
  const key = new THREE.DirectionalLight('#fff1dc', 0.8); key.position.set(-2, 9, 12); scene.add(key);
  mats = { spots: [] };
  for (const x of [4, 16]) {
    const spot = new THREE.SpotLight('#ff7a1a', 30, 30, 0.5, 0.6, 1.6); spot.position.set(x, 10, 6); spot.target.position.set(x + 2, 0, 0);
    scene.add(spot, spot.target); mats.spots.push(spot);
  }

  // The board runs on past the left edge of the screen and into the fog on the right.
  // Surfaces: dark glass (the default: "not huge on the fretboard look", picked "dark glass highway"), maple, dark wood.
  const white = new THREE.Color('#ffffff');
  mats.boards = { glass: { map: null, color: new THREE.Color('#0b0d13'), roughness: 0.55, metalness: 0.1 },   // satin: a mirror finish caught the lights ("too shiny")
                  maple: { map: woodTexture('#6f4f2a', 'rgba(220,185,130,1)', 'rgba(60,36,12,1)', 0.9), color: white, roughness: 0.7, metalness: 0 },
                  dark: { map: woodTexture('#1b1512', 'rgba(120,95,70,1)', 'rgba(0,0,0,1)', 0.5), color: white, roughness: 0.55, metalness: 0.25 } };
  board = new THREE.Mesh(new THREE.BoxGeometry(80, 0.3, 1), new THREE.MeshStandardMaterial(mats.boards.glass));
  world.add(board);
  mats.fret = new THREE.MeshStandardMaterial({ color: '#c9c3b8', metalness: 1, roughness: 0.35 });
  mats.line = new THREE.MeshBasicMaterial({ color: '#ffe9c9', transparent: true, opacity: 0.14 });
  mats.string = new THREE.MeshStandardMaterial({ color: '#8a847c', metalness: 0.6, roughness: 0.6 });
  mats.noteGeo = new RoundedBoxGeometry(0.86, 0.26, 0.78, 4, 0.12);
  mats.skirtGeo = new RoundedBoxGeometry(0.9, 0.14, 0.82, 3, 0.07);
  mats.numGeo = new THREE.PlaneGeometry(0.62, 0.62);
  mats.tailGeo = new THREE.BoxGeometry(1, 0.06, 0.16);
  mats.sparkGeo = new THREE.SphereGeometry(0.025, 6, 6);

  // The glow pass renders through its own targets, which had no antialiasing (jagged edges): 4x MSAA on them.
  composer = new EffectComposer(renderer, new THREE.WebGLRenderTarget(1, 1, { samples: 4, type: THREE.HalfFloatType }));
  composer.addPass(new RenderPass(scene, camera));
  composer.addPass(new UnrealBloomPass(new THREE.Vector2(1600, 1000), 0.35, 0.4, 0.98));
  composer.addPass(new OutputPass());
}

// Lights in the back (user: "maybe some sort of dynamic background ... to make it feel like a concert", then
// "no crowd, just lights in the back"): a soft glow far behind the board and out-of-focus stage lights drifting,
// brightening on the beat (more on beat 1); their colours follow the mood.
let show = null;
function gradientTexture(stops, vertical = true) {
  const c = document.createElement('canvas'); c.width = vertical ? 4 : 256; c.height = vertical ? 256 : 256; const g = c.getContext('2d');
  const grad = vertical ? g.createLinearGradient(0, 0, 0, 256) : g.createRadialGradient(128, 128, 0, 128, 128, 128);
  stops.forEach(([at, col]) => grad.addColorStop(at, col));
  g.fillStyle = grad; g.fillRect(0, 0, c.width, c.height);
  return new THREE.CanvasTexture(c);
}
function buildShow() {
  // Out-of-focus stage lights (bokeh; beams looked "cheap ... like a flash game"), made livelier ("just more
  // dynamic"): three rows like lighting rigs at different depths, each swaying; a wave chases across them at each
  // bar, rows flash in turn on the beats, your hits flare a light in the string's colour, and the whole show gets
  // brighter with the multiplier.
  const group = new THREE.Group(); scene.add(group);
  const soft = gradientTexture([[0, 'rgba(255,255,255,1)'], [0.35, 'rgba(255,255,255,0.55)'], [0.7, 'rgba(255,255,255,0.12)'], [1, 'rgba(255,255,255,0)']], false);
  const mat = (o) => new THREE.SpriteMaterial({ map: soft, transparent: true, opacity: o, blending: THREE.AdditiveBlending, depthWrite: false, fog: false, toneMapped: false });
  const wall = new THREE.Mesh(new THREE.PlaneGeometry(240, 120), new THREE.MeshBasicMaterial({ map: soft, transparent: true, opacity: 0.18,
    blending: THREE.AdditiveBlending, depthWrite: false, fog: false, toneMapped: false }));
  wall.position.set(14, 4, -80); group.add(wall);
  const rows = [];
  [[-48, 18, 11, 7], [-60, 10, 13, 9], [-72, 24, 15, 12]].forEach(([z, y, count, size], r) => {
    const rig = new THREE.Group(); rig.position.set(14, y, z); group.add(rig);
    const lights = [];
    for (let i = 0; i < count; i++) {
      const glow = new THREE.Sprite(mat(0)); const sz = size * (0.8 + Math.random() * 0.4);
      glow.scale.set(sz, sz, 1); glow.position.set((i - (count - 1) / 2) * (100 / count), 0, 0); rig.add(glow);
      lights.push({ glow, flare: 0, flareCol: new THREE.Color(), phase: Math.random() * 6.3 });
    }
    rows.push({ rig, lights, r });
  });
  show = { group, wall, rows, lastBurst: 0, lastBar: -1, chaseAt: -1e9 };
}
// Seconds since the last beat and whether it was beat 1, from the tab's bars and tempo.
function beatAt(bars, bpm, t, playing) {
  if (!playing || !bars?.length || t < bars[0]) return { beat: 9, down: false };
  let lo = 0, hi = bars.length - 1;
  while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (bars[mid] <= t) lo = mid; else hi = mid - 1; }
  const b0 = bars[lo], b1 = bars[lo + 1] ?? b0 + 240 / (bpm || 120);
  const beats = Math.max(1, Math.round(((b1 - b0) * (bpm || 120)) / 60)), step = (b1 - b0) / beats, k = Math.floor((t - b0) / step);
  return { beat: t - (b0 + k * step), down: k === 0 };
}
const SHOW_COLORS = [   // [a, b] per mood: struggling, steady, hot (x4)
  [new THREE.Color('#7f1d1d'), new THREE.Color('#52525b')],
  [new THREE.Color('#f59e0b'), new THREE.Color('#e11d48')],
  [new THREE.Color('#38bdf8'), new THREE.Color('#818cf8')],
];
function updateShow(ms, v, pulse) {
  if (!show) buildShow();
  const { beat, down } = beatAt(v.bars, v.bpm, v.t, v.playing);
  const flash = Math.exp(-beat * 7) * (down ? 1 : 0.55);
  const m = v.mood, k = m < 0.55 ? m / 0.55 : (m - 0.55) / 0.45, [from, to] = m < 0.55 ? [SHOW_COLORS[0], SHOW_COLORS[1]] : [SHOW_COLORS[1], SHOW_COLORS[2]];
  const ca = from[0].clone().lerp(to[0], k), cb = from[1].clone().lerp(to[1], k);
  const energy = (0.35 + 0.65 * Math.min(1, m / 0.55)) * (0.75 + 0.1 * (v.mult || 1));   // brighter with the multiplier
  // A new bar: a wave of light chases across the rigs.
  if (v.playing && v.bars?.length) {
    let bar = 0; while (bar + 1 < v.bars.length && v.bars[bar + 1] <= v.t) bar++;
    if (bar !== show.lastBar) { show.lastBar = bar; show.chaseAt = ms; }
  }
  const chase = (ms - show.chaseAt) / 700;               // 0..1 across the rigs
  // Your hits flare a light (front rig) in the string's colour.
  for (const b of v.bursts) {
    if (b.at <= show.lastBurst) continue;
    const row = show.rows[0], l = row.lights[Math.floor(Math.random() * row.lights.length)];
    l.flare = 1; l.flareCol.set(v.colors[b.s % v.colors.length]);
  }
  show.lastBurst = Math.max(show.lastBurst, ...v.bursts.map((b) => b.at));
  const beatNo = v.playing ? Math.floor(v.t / (60 / (v.bpm || 120))) : 0;
  show.wall.material.color.copy(ca);
  show.wall.material.opacity = (0.16 + 0.22 * flash) * energy + (v.onFire && down ? 0.25 * Math.exp(-beat * 10) : 0);
  for (const row of show.rows) {
    row.rig.rotation.z = Math.sin(ms * 0.00018 + row.r * 2.1) * 0.06;   // rigs swaying
    row.rig.position.x = 14 + Math.sin(ms * 0.00011 + row.r) * 6;
    const onBeat = beatNo % 3 === row.r;                  // the rows flash in turn
    row.lights.forEach((l, i) => {
      const across = i / Math.max(1, row.lights.length - 1), wave = Math.max(0, 1 - Math.abs(chase - across) * 5) * (chase < 1.2 ? 1 : 0);
      const twinkle = 0.75 + 0.25 * Math.sin(ms * 0.002 + l.phase);
      let o = (0.1 + 0.18 * (onBeat ? flash : flash * 0.3) + 0.35 * wave) * energy * twinkle;
      const col = (i + row.r) % 2 ? cb : ca;
      l.glow.material.color.copy(col);
      if (l.flare > 0.01) { l.glow.material.color.lerp(l.flareCol, l.flare); o += 0.22 * l.flare; l.flare *= 0.9; }   // soft: bright ones bloomed over the board
      if (v.onFire && down && beat < 0.08) { l.glow.material.color.lerp(new THREE.Color('#ffffff'), 0.4); o += 0.12; }   // x4: a white flash on beat 1
      l.glow.material.opacity = Math.min(0.5, o);
      const s = 1 + 0.25 * wave + 0.15 * (onBeat ? flash : 0);
      l.glow.scale.setScalar((l.size ||= l.glow.scale.x) * s);
    });
  }
}

const laneZ = (s) => ((lanes.n - 1) / 2 - s) * LANE;   // low string nearest the camera = bottom of the screen

// Strings, colour bands, pads and names for n strings (4 bass, 5 five-string, 6 guitar).
function buildLanes(n, colors, names) {
  for (const o of lanes.objects || []) world.remove(o);
  lanes = { n, objects: [], pads: [], names: [], bands: [], paint: [], moveLines: [] };
  const boardW = LANE * n + 0.4;
  board.scale.z = boardW; board.position.set(30, -0.15, 0);
  const add = (o) => { world.add(o); lanes.objects.push(o); return o; };
  for (let s = 0; s < n; s++) {
    const col = colors[s % colors.length];
    lanes.paint.push(new THREE.Color(col).multiplyScalar(0.55));   // note bodies: deeper, so light doesn't wash them out
    const band = add(new THREE.Mesh(new THREE.PlaneGeometry(80, LANE * 0.94), new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.1, depthWrite: false })));
    band.rotation.x = -Math.PI / 2; band.position.set(30, 0.002, laneZ(s)); band.userData.col = col; lanes.bands.push(band);
    const r = Math.max(0.018, 0.045 - s * (0.027 / Math.max(1, n - 1)));
    const str = add(new THREE.Mesh(new THREE.CylinderGeometry(r, r, 80, 10), new THREE.MeshStandardMaterial({ color: '#8a847c', metalness: 0.6, roughness: 0.5, emissive: col, emissiveIntensity: 0.18 }))); str.rotation.z = Math.PI / 2; str.position.set(30, 0.05, laneZ(s)); (lanes.strings ||= []).push(str);
    const rim = add(new THREE.Mesh(new RoundedBoxGeometry(NOTE_W + 0.16, 0.1, 0.9, 4, 0.08), new THREE.MeshStandardMaterial({ color: '#000', emissive: col, emissiveIntensity: 0.55 })));
    rim.position.set(0, 0.06, laneZ(s));
    const lens = add(new THREE.Mesh(new RoundedBoxGeometry(NOTE_W + 0.04, 0.12, 0.78, 4, 0.08), new THREE.MeshPhysicalMaterial({ color: '#100c0a', roughness: 0.2, clearcoat: 1, emissive: col, emissiveIntensity: 0.12 })));
    lens.position.set(0, 0.13, laneZ(s));
    const light = add(new THREE.PointLight(col, 0, 4, 2)); light.position.set(0.3, 0.8, laneZ(s));
    const name = add(new THREE.Sprite(new THREE.SpriteMaterial({ map: label(names[s] || '', col, true, 128, true), depthWrite: false })));
    name.scale.set(0.62, 0.62, 1); name.position.set(NUT - 0.5, 0.35, laneZ(s));
    name.visible = false;                                // drawn crisply on the HUD instead (nameScreen); kept for the camera fit
    lanes.pads.push({ rim, lens, light, col, hitAt: -1e9 });
    lanes.names.push(name);
  }
  lanes.boardW = boardW;
  lanes.edges = [-1, 1].map((side) => {                  // glass: a thin light edge along each side
    const e = add(new THREE.Mesh(new THREE.BoxGeometry(80, 0.03, 0.03), new THREE.MeshBasicMaterial({ color: '#fff5e6', transparent: true, opacity: 0.35, toneMapped: false })));
    e.position.set(30, 0.01, side * (boardW / 2)); return e;
  });
  lanes.style = null;                                    // re-apply the surface to the new strings and bands
  // At x4 the highway turns blue (user: "make the highway blue too"): a blue wash over the board,
  // strongest at the strike line, faded in and out by frame().
  const c = document.createElement('canvas'); c.width = 512; c.height = 4; const g = c.getContext('2d');
  const grad = g.createLinearGradient(0, 0, 512, 0); grad.addColorStop(0, 'rgba(255,255,255,0)'); grad.addColorStop(0.08, 'rgba(255,255,255,1)'); grad.addColorStop(1, 'rgba(255,255,255,0.25)');
  g.fillStyle = grad; g.fillRect(0, 0, 512, 4);
  const washTex = new THREE.CanvasTexture(c);
  lanes.wash = add(new THREE.Mesh(new THREE.PlaneGeometry(40, boardW), new THREE.MeshBasicMaterial({ color: '#1557c0', map: washTex, transparent: true, opacity: 0, depthWrite: false })));
  lanes.wash.rotation.x = -Math.PI / 2; lanes.wash.position.set(NUT - 1 + 20, 0.004, 0); lanes.fire = 0;
  for (const b of bars) world.remove(b.grp);
  bars = [];
}

// A bar line: a nickel fret wire (maple) or a faint light line (dark). No dot inlays (user: "remove the dot").
function barObject() {
  const grp = new THREE.Group();
  const wire = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, lanes.boardW, 10), mats.fret); wire.rotation.x = Math.PI / 2; wire.position.y = 0.01;
  const line = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.014, lanes.boardW), mats.line); line.position.y = 0.008;
  const num = new THREE.Sprite(new THREE.SpriteMaterial({ depthWrite: false, transparent: true }));   // the bar's number, above the top string
  num.scale.set(0.42, 0.42, 1); num.position.set(0.28, 0.12, laneZ(lanes.n - 1) - LANE * 0.55);
  grp.add(wire, line, num); world.add(grp);
  return { grp, wire, line, num, shown: -1 };
}

// A note from the pool: a glossy body on a dark skirt, the number printed on top, a tail for long notes.
function noteObject() {
  const grp = new THREE.Group();
  const body = new THREE.Mesh(mats.noteGeo, new THREE.MeshPhysicalMaterial({ roughness: 0.4, clearcoat: 0.6, clearcoatRoughness: 0.35, transparent: true }));
  body.position.y = 0.2;
  const base = new THREE.Mesh(mats.skirtGeo, new THREE.MeshStandardMaterial({ roughness: 0.6, transparent: true }));
  base.position.y = 0.07;
  const num = new THREE.Mesh(mats.numGeo, new THREE.MeshBasicMaterial({ transparent: true, depthWrite: false, toneMapped: false }));
  num.rotation.x = -Math.PI / 2; num.position.y = 0.335;
  const tail = new THREE.Mesh(mats.tailGeo, new THREE.MeshStandardMaterial({ transparent: true, roughness: 0.5 }));
  tail.position.y = 0.06;
  const tailGlow = new THREE.Mesh(mats.tailGeo, new THREE.MeshBasicMaterial({ transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false }));
  tailGlow.visible = false;                              // the soft glow round a held note's tail
  grp.add(base, body, num); world.add(grp); world.add(tail); world.add(tailGlow);
  return { grp, body, base, num, tail, tailGlow, k: -1, sinkAt: 0 };
}

function setFretboard(style) {
  if (lanes.style === style) return;
  lanes.style = style;
  Object.assign(board.material, mats.boards[style] || mats.boards.glass); board.material.needsUpdate = true;
  lanes.bands.forEach((b, s) => { b.material.opacity = style === 'glass' ? (s % 2 ? 0.035 : 0.022) : style === 'dark' ? (s % 2 ? 0.07 : 0.045) : 0.1; });
  lanes.strGlow = style === 'glass' ? 0.6 : 0.18;      // glass: the strings are thin glowing lines
  lanes.edges?.forEach((e) => { e.visible = style === 'glass'; });
}

// The camera: in front, turned 5 degrees right ("don't make it so angled"), slid sideways until every
// string name sits inside the left edge, measured on screen (a wide window pushed them off).
function aim(w, h) {
  const aspect = w / h, z = 16, yaw = (5 * Math.PI) / 180;
  camera.fov = aspect < 1.2 ? 50 : 40; camera.aspect = aspect; camera.updateProjectionMatrix();
  const half = Math.atan(Math.tan((camera.fov * Math.PI) / 360) * aspect);
  let x = NUT - 1.2 + z * Math.tan(half - yaw);
  const place = () => { camera.position.set(x, 1.0 + (lanes.n - 4) * 0.4, z + (lanes.n - 4) * 1.6); camera.lookAt(x + z * Math.tan(yaw), -0.5, 0); camera.updateMatrixWorld(); };
  place(); world.updateMatrixWorld();
  for (let i = 0; i < 30; i++) {
    const left = Math.min(...lanes.names.map((s) => s.getWorldPosition(v3).project(camera).x));
    if (left >= -0.93) break;
    x -= (-0.93 - left) * 3 + 0.05; place();
  }
}

const mix = (a, b, k) => a.clone().lerp(b, k);
const MOOD = {   // background / spot colours: struggling, steady, hot (x4)
  bg: [new THREE.Color('#070607'), new THREE.Color('#0b0806'), new THREE.Color('#040a14')],
  spotA: [new THREE.Color('#7f1d1d'), new THREE.Color('#ff7a1a'), new THREE.Color('#38bdf8')],
  spotB: [new THREE.Color('#3f3f46'), new THREE.Color('#e11d48'), new THREE.Color('#818cf8')],
};
const moodColor = (list, m) => (m < 0.55 ? mix(list[0], list[1], m / 0.55) : mix(list[1], list[2], (m - 0.55) / 0.45));

export function mount(wrap) {
  if (!renderer) init();
  wrapEl = wrap;
  wrap.prepend(renderer.domElement);
  lastSize = '';
}

// Where string s's name goes on screen (CSS px within the wrap): the HUD draws it there, crisp (as a texture in the
// scene the letters looked "sloppy").
export function nameScreen(s) {
  const n = lanes.names?.[s];
  if (!n || !wrapEl) return null;
  n.getWorldPosition(v3).project(camera);
  return { x: ((v3.x + 1) / 2) * wrapEl.clientWidth, y: ((1 - v3.y) / 2) * wrapEl.clientHeight };
}

// Where pad s is on screen (CSS px within the wrap), for the HUD's calls.
export function padScreen(s) {
  const p = lanes.pads?.[s];                             // none until the first frame built the strings
  if (!p || !wrapEl) return null;
  p.lens.getWorldPosition(v3).project(camera);
  return { x: ((v3.x + 1) / 2) * wrapEl.clientWidth, y: ((1 - v3.y) / 2) * wrapEl.clientHeight };
}

// One frame. v: { t, notes, n, colors, names, Z (seconds ahead), quickGap, prev, result, laneHitMs,
// laneMissMs, sustain, bursts, onFire, mood (0-1), fretboard, bars, bpm, inputLane, legato(k) }
export function frame(v) {
  const w = wrapEl.clientWidth, h = wrapEl.clientHeight, ms = performance.now();
  if (!w || !h) return;
  if (lanes.n !== v.n || lanes.key !== v.names.join()) { buildLanes(v.n, v.colors, v.names); lanes.key = v.names.join(); lastSize = ''; }
  setFretboard(v.fretboard);
  if (lastSize !== `${w}x${h}`) {
    lastSize = `${w}x${h}`;
    renderer.setSize(w, h, false); composer.setSize(w, h);
    renderer.domElement.style.width = `${w}px`; renderer.domElement.style.height = `${h}px`;
    aim(w, h);
  }

  // Speed: the note speed setting's seconds ahead across the visible board, but fast enough that
  // distinct notes on a string never touch (quick same-fret repeats are ticks, so they don't count).
  const speed = Math.min(1.6 * (VISIBLE / v.Z), Math.max(VISIBLE / v.Z, (NOTE_W + GAP) / Math.max(0.05, v.quickGap)));
  const far = speed * v.Z, t = v.t, notes = v.notes;

  // Mood: background, fog and the two stage spots ease between struggling, steady and hot.
  const lastHit = Math.max(-1e9, ...Object.values(v.laneHitMs)), pulse = Math.exp(-(ms - lastHit) / 180);
  const bg = moodColor(MOOD.bg, v.mood);
  renderer.setClearColor(bg, 1); scene.fog.color.copy(bg);
  mats.spots[0].color.copy(moodColor(MOOD.spotA, v.mood)); mats.spots[1].color.copy(moodColor(MOOD.spotB, v.mood));
  const glow = (0.35 + 0.65 * Math.min(1, v.mood / 0.55)) * (v.onFire ? 1.2 + 0.5 * pulse : 1);
  mats.spots[0].intensity = 30 * glow; mats.spots[1].intensity = 25 * glow;

  // The blue highway at x4, easing in and out (about a third of a second), pulsing with each hit.
  lanes.fire += ((v.onFire ? 1 : 0) - lanes.fire) * Math.min(1, (ms - (lanes.lastMs || ms)) / 300);
  lanes.lastMs = ms;
  lanes.wash.material.opacity = lanes.fire * (0.55 + 0.1 * pulse);
  lanes.bands.forEach((b) => b.material.color.lerpColors(new THREE.Color(b.userData.col), new THREE.Color('#38bdf8'), lanes.fire));

  updateShow(ms, v, pulse);

  // Bar lines.
  const visibleBars = [];
  for (let i = 0; i < v.bars.length; i++) { const x = (v.bars[i] - t) * speed; if (x > far + 1) break; if (x > NUT - 0.2) visibleBars.push([i, x]); }
  while (bars.length < visibleBars.length) bars.push(barObject());
  bars.forEach((b, j) => {
    const vb = visibleBars[j];
    b.grp.visible = !!vb;
    if (!vb) return;
    const [i, x] = vb, maple = v.fretboard !== 'dark';
    b.grp.position.set(x, 0, 0);
    b.wire.visible = maple && v.fretboard === 'maple'; b.line.visible = !b.wire.visible;   // fret wires on maple, soft light lines otherwise
    if (b.shown !== i) { b.shown = i; b.num.material.map = tex[`bar${i}`] ||= label(String(i + 1), '#d6c7b0', true); b.num.material.needsUpdate = true; }
    b.num.material.opacity = Math.max(0, Math.min(1, (far - x) / FADE)) * (x > 0.6 ? 0.8 : 0);
  });

  // Hand moves: a dashed amber line across the board just before the hand moves along the neck.
  const moves = (v.moves || []).map((m) => (m - t) * speed).filter((x) => x > 0.3 && x < far);
  while (lanes.moveLines.length < moves.length) {
    const c = document.createElement('canvas'); c.width = 4; c.height = 64; const g = c.getContext('2d');
    g.fillStyle = '#fbbf24'; for (let y = 0; y < 64; y += 16) g.fillRect(0, y, 4, 9);
    const t2 = new THREE.CanvasTexture(c); t2.wrapT = THREE.RepeatWrapping; t2.repeat.set(1, lanes.n * 2);
    const line = new THREE.Mesh(new THREE.PlaneGeometry(0.07, lanes.boardW - 0.2), new THREE.MeshBasicMaterial({ map: t2, transparent: true, depthWrite: false, toneMapped: false }));
    line.rotation.x = -Math.PI / 2; line.position.y = 0.012; world.add(line); lanes.objects.push(line); lanes.moveLines.push(line);
  }
  lanes.moveLines.forEach((l, j) => { l.visible = j < moves.length; if (l.visible) { l.position.x = moves[j]; l.material.opacity = 0.85 * Math.max(0, Math.min(1, (far - moves[j]) / FADE)); } });

  // Notes on the board now (past ones linger briefly to sink into the pad or slide off).
  let lo = 0, hi = notes.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (notes[mid][0] < t - 2) lo = mid + 1; else hi = mid; }
  const live = new Set();
  for (let k = lo; k < notes.length && (notes[k][0] - t) * speed < far; k++) live.add(k);
  for (const [k, o] of used) if (!live.has(k)) { o.grp.visible = false; o.tail.visible = false; o.k = -1; pool.push(o); used.delete(k); }
  const held = new Set(), heldEnds = new Set();         // strings with a note held now / ending now
  for (const k of live) {
    const [ns, ne, s, f] = notes[k];
    if (s >= lanes.n) continue;
    let o = used.get(k);
    const res = v.result[k], missed = res === 'miss', open = f === 0;
    // The first note has its number; the same fret again on that string is the thin bar (user: "I wanted the first
    // note to have a number and any repeating to have the thinner bar"); after a rest of 2.5 s it's a full note again.
    // A repeat is the very next note played, same string and fret, nothing in between (A5 E0 A5 isn't one: the user
    // read those as new notes, "there's not repeating notes that are not showing").
    let pk = k - 1;
    while (pk >= 0 && ns - notes[pk][0] < 0.03 && notes[pk][2] !== s) pk--;   // skip notes struck together with it
    const tick = pk >= 0 && notes[pk][2] === s && notes[pk][3] === f && ns - notes[pk][0] < 2.5, repeat = false;
    if (!o) {
      o = pool.pop() || noteObject(); used.set(k, o); o.k = k; o.sinkAt = 0; o.style = '';
    }
    const style = `${s}|${f}|${missed}|${tick}|${repeat}|${v.onFire}`;
    if (o.style !== style) {                            // (re)paint for this note
      o.style = style;
      const col = new THREE.Color(v.colors[s % v.colors.length]);
      o.body.material.color.copy(missed ? new THREE.Color('#3f3f46') : open ? new THREE.Color('#120e0b') : lanes.paint[s]);
      o.body.material.emissive.copy(v.onFire && !missed ? new THREE.Color('#38bdf8') : col); o.body.material.emissiveIntensity = missed ? 0 : v.onFire ? 0.08 : open ? 0.06 : 0;
      o.base.material.color.copy(missed ? new THREE.Color('#18181b') : open ? col : new THREE.Color('#0a0705'));
      o.base.material.emissive.copy(col); o.base.material.emissiveIntensity = open && !missed ? 0.8 : 0;
      const key = `${f}|${open}|${s}|${missed}`;
      tex[key] ||= label(String(f), missed ? '#a1a1aa' : open ? v.colors[s % v.colors.length] : '#ffffff', !open && !missed);
      o.num.material.map = tex[key]; o.num.material.needsUpdate = true;
      o.num.visible = !tick && !repeat;
      const wide = tick ? 0.14 / 0.86 : NOTE_W / 0.86;
      o.body.scale.x = o.base.scale.x = wide;
      o.tail.material.color.copy(missed ? new THREE.Color('#3f3f46') : col);
    }
    // Where it is: hit notes sink into the pad (90 ms); held long notes sit on the pad.
    let end = Math.max(ne, ns);
    for (let j = k + 1; j < notes.length && notes[j][0] < end; j++) if (notes[j][2] === s) { end = notes[j][0]; break; }
    const holding = res === 'hit' && ns <= t && t < end && end - ns > 0.15;
    if (res === 'hit' && !o.sinkAt) o.sinkAt = ms;
    const sink = res === 'hit' && !holding ? Math.min(1, (ms - o.sinkAt) / 90) : 0;
    const x = res === 'hit' ? 0 : (ns - t) * speed;
    const come = Math.max(0, Math.min(1, (far - x) / FADE)), ease = come * come * (3 - 2 * come);   // fading in from the distance
    o.grp.visible = (res !== 'hit' || holding || sink < 1) && x > NUT + 0.2 && x < far;
    o.grp.position.set(x, 0.35 * (1 - ease) - 0.25 * sink, laneZ(s));
    o.grp.scale.setScalar((1 - 0.5 * sink) * (0.85 + 0.15 * ease));
    const fade = ease * (missed ? 0.7 : 1);
    o.body.material.opacity = o.base.material.opacity = fade; o.num.material.opacity = fade;
    // A slim tail as long as the note rings.
    const x0 = Math.max(0, x), x1 = Math.min(far, (end - t) * speed);
    o.tail.visible = end - ns > 0.15 && x1 > x0 + 0.1 && o.grp.visible;
    // Holding (user: "make the effect of holding a note look better"): the tail burns bright with a soft glow
    // round it and drains into the pad, the note glows on the pad, pulsing.
    const throb = 0.5 + 0.5 * Math.sin(ms / 70);
    if (o.tail.visible) {
      o.tail.scale.x = x1 - x0; o.tail.position.set((x0 + x1) / 2, 0.06, laneZ(s));
      o.tail.material.opacity = holding ? 0.95 : 0.55 * fade; o.tail.material.emissive.copy(o.tail.material.color);
      o.tail.material.emissiveIntensity = holding ? 1.1 + 0.5 * throb : 0;
    }
    o.tailGlow.visible = o.tail.visible && holding;
    if (o.tailGlow.visible) {
      o.tailGlow.scale.set(x1 - x0, 1.6, 3.2); o.tailGlow.position.set((x0 + x1) / 2, 0.07, laneZ(s));
      o.tailGlow.material.color.copy(v.onFire ? new THREE.Color('#38bdf8') : new THREE.Color(v.colors[s % v.colors.length]));
      o.tailGlow.material.opacity = 0.28 + 0.14 * throb;
    }
    if (holding) { o.body.material.emissiveIntensity = 0.45 + 0.3 * throb; held.add(s); if (end - t < 0.06) heldEnds.add(s); }
    else if (o.style.endsWith('false') && !missed && !open) o.body.material.emissiveIntensity = 0;
  }

  // Pads: a hit dips the pad and lights its lens and a pool of light; at x4 everything goes blue;
  // what you're playing now outlines its pad in white.
  for (const [s, p] of lanes.pads.entries()) {
    const at = v.laneHitMs[s] ?? -1e9, age = ms - at, k = Math.max(0, 1 - age / 220), dip = age < 60 ? age / 60 : Math.max(0, 1 - (age - 60) / 120);
    const bad = Math.max(0, 1 - (ms - (v.laneMissMs[s] ?? -1e9)) / 300), held = ms - (v.sustain[s] ?? -1e9) < 120;
    const col = v.onFire ? '#38bdf8' : p.col;
    p.lens.material.emissive.set(bad > k ? '#ef4444' : col); p.lens.material.emissiveIntensity = 0.12 + 1.5 * Math.max(k, bad * 0.6, held ? 0.6 + 0.3 * Math.sin(ms / 70) : 0);
    p.light.color.set(col); p.light.intensity = 4 * Math.max(k, held ? 0.5 : 0);
    p.lens.position.y = 0.13 - 0.06 * dip; p.rim.position.y = 0.06 - 0.04 * dip;
    p.rim.material.emissive.set(v.inputLane === s ? '#ffffff' : bad > 0.3 ? '#ef4444' : col);
    p.rim.material.emissiveIntensity = v.inputLane === s ? 1.1 : 0.55;
  }
  // Held strings vibrate and glow; sparks stream off the pad; a small burst when the hold reaches its end.
  lanes.strings?.forEach((str, s) => {
    const on = held.has(s);
    str.position.y = 0.05 + (on ? Math.sin(ms / 16 + s) * 0.018 : 0);
    str.material.emissiveIntensity = on ? 0.9 + 0.3 * Math.sin(ms / 50) : lanes.strGlow ?? 0.18;
  });
  const spark = (s, count, up) => {
    const cols = v.onFire ? ['#e0f2fe', '#38bdf8'] : [lanes.pads[s].col, '#ffe7a3'];
    for (let i = 0; i < count; i++) {
      const sp = new THREE.Mesh(mats.sparkGeo, new THREE.MeshBasicMaterial({ color: cols[i % 2], transparent: true, toneMapped: false }));
      sp.position.set(0, 0.3, laneZ(s)); world.add(sp);
      sparks.push({ sp, z0: laneZ(s), v: [0.3 + Math.random() * up, 1.6 + Math.random() * up * 1.5, (Math.random() - 0.5) * 0.8], born: ms });
    }
  };
  for (const s of held) if (Math.random() < 0.45) spark(s, 1, 0.8);
  for (const s of heldEnds) if (!lanes.pads[s].burstAt || ms - lanes.pads[s].burstAt > 400) { lanes.pads[s].burstAt = ms; spark(s, 8, 1.6); }

  // New hits: a few small sparks rising off the pad.
  for (const b of v.bursts) {
    if (b.at <= seenBurst || b.s >= lanes.n) continue;
    const cols = v.onFire ? ['#e0f2fe', '#38bdf8', '#7dd3fc', '#e0f2fe'] : [lanes.pads[b.s].col, '#ffe7a3', lanes.pads[b.s].col, lanes.pads[b.s].col];
    for (let i = 0; i < 4; i++) {
      const sp = new THREE.Mesh(mats.sparkGeo, new THREE.MeshBasicMaterial({ color: cols[i], transparent: true, toneMapped: false }));
      sp.position.set(0, 0.3, laneZ(b.s)); world.add(sp);
      sparks.push({ sp, z0: laneZ(b.s), v: [0.6 + Math.random() * 1.2, 2.2 + Math.random() * 1.4, (Math.random() - 0.5) * 1.2], born: ms });
    }
  }
  seenBurst = Math.max(seenBurst, ...v.bursts.map((b) => b.at));
  for (let i = sparks.length - 1; i >= 0; i--) {
    const sk = sparks[i], a = (ms - sk.born) / 1000;
    if (a > 0.25) { world.remove(sk.sp); sk.sp.material.dispose(); sparks.splice(i, 1); continue; }
    sk.sp.position.set(sk.v[0] * a, 0.3 + sk.v[1] * a - 6 * a * a, sk.z0 + sk.v[2] * a); sk.sp.material.opacity = 1 - a / 0.25;
  }

  composer.render();
}

window.Game3D = { mount, frame, padScreen, nameScreen };
