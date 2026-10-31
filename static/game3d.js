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

function label(text, color, outline = false, size = 128) {
  const c = document.createElement('canvas'); c.width = c.height = size; const g = c.getContext('2d');
  g.font = `900 ${size * 0.74}px ${FONT}`; g.textAlign = 'center'; g.textBaseline = 'middle';
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
  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(2, devicePixelRatio));
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.domElement.className = 'game3d-gl';
  scene = new THREE.Scene();
  scene.background = new THREE.Color('#0b0806');
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
  mats.boards = { maple: { map: woodTexture('#6f4f2a', 'rgba(220,185,130,1)', 'rgba(60,36,12,1)', 0.9), roughness: 0.7, metalness: 0 },
                  dark: { map: woodTexture('#1b1512', 'rgba(120,95,70,1)', 'rgba(0,0,0,1)', 0.5), roughness: 0.55, metalness: 0.25 } };
  board = new THREE.Mesh(new THREE.BoxGeometry(80, 0.3, 1), new THREE.MeshStandardMaterial(mats.boards.maple));
  world.add(board);
  mats.fret = new THREE.MeshStandardMaterial({ color: '#c9c3b8', metalness: 1, roughness: 0.35 });
  mats.line = new THREE.MeshBasicMaterial({ color: '#ffe9c9', transparent: true, opacity: 0.14 });
  mats.string = new THREE.MeshStandardMaterial({ color: '#8a847c', metalness: 0.6, roughness: 0.6 });
  mats.noteGeo = new RoundedBoxGeometry(0.86, 0.26, 0.78, 4, 0.12);
  mats.skirtGeo = new RoundedBoxGeometry(0.9, 0.14, 0.82, 3, 0.07);
  mats.numGeo = new THREE.PlaneGeometry(0.62, 0.62);
  mats.tailGeo = new THREE.BoxGeometry(1, 0.06, 0.16);
  mats.sparkGeo = new THREE.SphereGeometry(0.025, 6, 6);

  composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  composer.addPass(new UnrealBloomPass(new THREE.Vector2(1600, 1000), 0.35, 0.4, 0.98));
  composer.addPass(new OutputPass());
}

const laneZ = (s) => ((lanes.n - 1) / 2 - s) * LANE;   // low string nearest the camera = bottom of the screen

// Strings, colour bands, pads and names for n strings (4 bass, 5 five-string, 6 guitar).
function buildLanes(n, colors, names) {
  for (const o of lanes.objects || []) world.remove(o);
  lanes = { n, objects: [], pads: [], names: [], bands: [], paint: [] };
  const boardW = LANE * n + 0.4;
  board.scale.z = boardW; board.position.set(30, -0.15, 0);
  const add = (o) => { world.add(o); lanes.objects.push(o); return o; };
  for (let s = 0; s < n; s++) {
    const col = colors[s % colors.length];
    lanes.paint.push(new THREE.Color(col).multiplyScalar(0.55));   // note bodies: deeper, so light doesn't wash them out
    const band = add(new THREE.Mesh(new THREE.PlaneGeometry(80, LANE * 0.94), new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.1, depthWrite: false })));
    band.rotation.x = -Math.PI / 2; band.position.set(30, 0.002, laneZ(s)); band.userData.col = col; lanes.bands.push(band);
    const r = Math.max(0.018, 0.045 - s * (0.027 / Math.max(1, n - 1)));
    const str = add(new THREE.Mesh(new THREE.CylinderGeometry(r, r, 80, 10), mats.string)); str.rotation.z = Math.PI / 2; str.position.set(30, 0.05, laneZ(s));
    const rim = add(new THREE.Mesh(new RoundedBoxGeometry(NOTE_W + 0.16, 0.1, 0.9, 4, 0.08), new THREE.MeshStandardMaterial({ color: '#000', emissive: col, emissiveIntensity: 0.55 })));
    rim.position.set(0, 0.06, laneZ(s));
    const lens = add(new THREE.Mesh(new RoundedBoxGeometry(NOTE_W + 0.04, 0.12, 0.78, 4, 0.08), new THREE.MeshPhysicalMaterial({ color: '#100c0a', roughness: 0.2, clearcoat: 1, emissive: col, emissiveIntensity: 0.12 })));
    lens.position.set(0, 0.13, laneZ(s));
    const light = add(new THREE.PointLight(col, 0, 4, 2)); light.position.set(0.3, 0.8, laneZ(s));
    const name = add(new THREE.Sprite(new THREE.SpriteMaterial({ map: label(names[s] || '', col, true), depthWrite: false })));
    name.scale.set(0.45, 0.45, 1); name.position.set(NUT - 0.45, 0.35, laneZ(s));
    lanes.pads.push({ rim, lens, light, col, hitAt: -1e9 });
    lanes.names.push(name);
  }
  lanes.boardW = boardW;
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
  grp.add(wire, line); world.add(grp);
  return { grp, wire, line };
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
  grp.add(base, body, num); world.add(grp); world.add(tail);
  return { grp, body, base, num, tail, k: -1, sinkAt: 0 };
}

function setFretboard(style) {
  if (lanes.style === style) return;
  lanes.style = style;
  Object.assign(board.material, mats.boards[style]); board.material.needsUpdate = true;
  lanes.bands.forEach((b, s) => { b.material.opacity = style === 'dark' ? (s % 2 ? 0.07 : 0.045) : 0.1; });
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
  scene.background.copy(bg); scene.fog.color.copy(bg);
  mats.spots[0].color.copy(moodColor(MOOD.spotA, v.mood)); mats.spots[1].color.copy(moodColor(MOOD.spotB, v.mood));
  const glow = (0.35 + 0.65 * Math.min(1, v.mood / 0.55)) * (v.onFire ? 1.2 + 0.5 * pulse : 1);
  mats.spots[0].intensity = 30 * glow; mats.spots[1].intensity = 25 * glow;

  // The blue highway at x4, easing in and out (about a third of a second), pulsing with each hit.
  lanes.fire += ((v.onFire ? 1 : 0) - lanes.fire) * Math.min(1, (ms - (lanes.lastMs || ms)) / 300);
  lanes.lastMs = ms;
  lanes.wash.material.opacity = lanes.fire * (0.55 + 0.1 * pulse);
  lanes.bands.forEach((b) => b.material.color.lerpColors(new THREE.Color(b.userData.col), new THREE.Color('#38bdf8'), lanes.fire));

  // Bar lines.
  const visibleBars = [];
  for (let i = 0; i < v.bars.length; i++) { const x = (v.bars[i] - t) * speed; if (x > far + 1) break; if (x > NUT - 0.2) visibleBars.push([i, x]); }
  while (bars.length < visibleBars.length) bars.push(barObject());
  bars.forEach((b, j) => {
    const vb = visibleBars[j];
    b.grp.visible = !!vb;
    if (!vb) return;
    const [, x] = vb, maple = v.fretboard !== 'dark';
    b.grp.position.set(x, 0, 0);
    b.wire.visible = maple; b.line.visible = !maple;
  });

  // Notes on the board now (past ones linger briefly to sink into the pad or slide off).
  let lo = 0, hi = notes.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (notes[mid][0] < t - 2) lo = mid + 1; else hi = mid; }
  const live = new Set();
  for (let k = lo; k < notes.length && (notes[k][0] - t) * speed < far; k++) live.add(k);
  for (const [k, o] of used) if (!live.has(k)) { o.grp.visible = false; o.tail.visible = false; o.k = -1; pool.push(o); used.delete(k); }
  for (const k of live) {
    const [ns, ne, s, f] = notes[k];
    if (s >= lanes.n) continue;
    let o = used.get(k);
    const res = v.result[k], missed = res === 'miss', open = f === 0;
    const pk = v.prev[k], tick = pk >= 0 && notes[pk][3] === f && (ns - notes[pk][0]) * speed < NOTE_W + GAP;
    if (!o) {
      o = pool.pop() || noteObject(); used.set(k, o); o.k = k; o.sinkAt = 0; o.style = '';
    }
    const style = `${s}|${f}|${missed}|${tick}|${v.onFire}`;
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
      o.num.visible = !tick;
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
    if (o.tail.visible) { o.tail.scale.x = x1 - x0; o.tail.position.set((x0 + x1) / 2, 0.06, laneZ(s)); o.tail.material.opacity = 0.55 * fade * (holding ? 1.5 : 1); o.tail.material.emissive.copy(o.tail.material.color); o.tail.material.emissiveIntensity = holding ? 0.6 : 0; }
  }

  // Pads: a hit dips the pad and lights its lens and a pool of light; at x4 everything goes blue;
  // what you're playing now outlines its pad in white.
  for (const [s, p] of lanes.pads.entries()) {
    const at = v.laneHitMs[s] ?? -1e9, age = ms - at, k = Math.max(0, 1 - age / 220), dip = age < 60 ? age / 60 : Math.max(0, 1 - (age - 60) / 120);
    const bad = Math.max(0, 1 - (ms - (v.laneMissMs[s] ?? -1e9)) / 300), held = ms - (v.sustain[s] ?? -1e9) < 120;
    const col = v.onFire ? '#38bdf8' : p.col;
    p.lens.material.emissive.set(bad > k ? '#ef4444' : col); p.lens.material.emissiveIntensity = 0.12 + 1.5 * Math.max(k, bad * 0.6, held ? 0.7 : 0);
    p.light.color.set(col); p.light.intensity = 4 * Math.max(k, held ? 0.5 : 0);
    p.lens.position.y = 0.13 - 0.06 * dip; p.rim.position.y = 0.06 - 0.04 * dip;
    p.rim.material.emissive.set(v.inputLane === s ? '#ffffff' : bad > 0.3 ? '#ef4444' : col);
    p.rim.material.emissiveIntensity = v.inputLane === s ? 1.1 : 0.55;
  }
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

window.Game3D = { mount, frame, padScreen };
