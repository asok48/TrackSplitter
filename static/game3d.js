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

const LANE = 1.5, NUT = -0.62, NOTE_W = 0.56, GAP = 0.16, VISIBLE = 17.5, FADE = 1.8;   // fade-in was 3.5: far notes stayed faint ("not enough warning")
const FONT = '"Avenir Next Condensed", Impact, sans-serif';

let renderer, composer, scene, camera, world, board, mats, wrapEl = null;
let lanes = { n: 0 }, bars = [], pool = [], used = new Map(), sparks = [], seenBurst = 0, lastSize = '';
const tex = {};
const v3 = new THREE.Vector3();
const BLUE = new THREE.Color('#38bdf8'), WHITE = new THREE.Color('#ffffff'), GOLD = new THREE.Color('#ffd58a');

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
  renderer.setPixelRatio(pace.pr);                       // the screen's full sharpness, or what settled last time (keepPace)
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.domElement.className = 'game3d-gl';
  scene = new THREE.Scene();
  scene.background = null;
  scene.fog = new THREE.Fog('#0b0806', 15, 36);
  scene.environment = new THREE.PMREMGenerator(renderer).fromScene(new RoomEnvironment(), 0.04).texture;
  scene.environmentIntensity = 0.3;
  // Far plane 300: the glow far behind (z -80, ~96 from the camera) crossed 100 and was cut off along a straight
  // diagonal, a dark wedge in the bottom corner (user: "a weird dark static triangle going across the screen").
  camera = new THREE.PerspectiveCamera(40, 16 / 10, 0.1, 300);
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
  mats.capGeo = new RoundedBoxGeometry(0.1, 0.12, 0.5, 3, 0.04);
  mats.reflGeo = new THREE.PlaneGeometry(1, 1);
  mats.sparkGeo = new THREE.SphereGeometry(0.025, 6, 6);

  // The glow pass renders through its own targets, which had no antialiasing (jagged edges): 4x MSAA on them.
  composer = new EffectComposer(renderer, new THREE.WebGLRenderTarget(1, 1, { samples: 4, type: THREE.HalfFloatType }));
  composer.addPass(new RenderPass(scene, camera));
  // The glow at a quarter of the screen's pixels (three.js uses half): it's a soft blur either way, and the game
  // was limited by pixels drawn (user: "can we also improve performance?"; a quarter of the pixels doubled the fps).
  const bloom = new UnrealBloomPass(new THREE.Vector2(1600, 1000), 0.35, 0.4, 0.98), bloomSize = bloom.setSize.bind(bloom);
  bloom.setSize = (w, h) => bloomSize(w / 2, h / 2);
  composer.addPass(bloom); mats.bloom = bloom;   // big moments swell the glow
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
  show = { group, wall, rows, soft, lastBurst: 0, lastBar: -1, chaseAt: -1e9 };
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
// Light patterns (user: "make the lights more dynamic instead of going left to right everytime"): one per bar, never
// the same twice running, picked to suit how loud the song is there. Each gives a light's brightness from where it is
// (`at`: 0-1 across its rig, rig 0 at the front) and the moment: `go` 0-1 through the bar's first 0.7 s (the moving
// ones), `hit` the latest kick / snare and `no` which hit that was (the ones that flash on hits).
const wave = (go, pos) => Math.max(0, 1 - Math.abs(go * 1.25 - pos) * 5) * (go < 1.2 ? 1 : 0);
const hash = (a, b) => { const x = Math.sin(a * 127.1 + b * 311.7) * 43758.5453; return x - Math.floor(x); };
const PATTERNS = {
  sweepRight: { calm: 1, loud: 1, light: (at, r, c) => wave(c.go, at) },
  sweepLeft: { calm: 1, loud: 1, light: (at, r, c) => wave(c.go, 1 - at) },
  fromCenter: { calm: 0.7, loud: 1.2, light: (at, r, c) => wave(c.go, Math.abs(at - 0.5) * 2) },
  toCenter: { calm: 0.7, loud: 1, light: (at, r, c) => wave(c.go, 1 - Math.abs(at - 0.5) * 2) },
  frontToBack: { calm: 0.8, loud: 1, light: (at, r, c) => wave(c.go, r / 2) },
  sparkle: { calm: 0, loud: 0.8, light: (at, r, c, i) => (hash(i + r * 31, c.no) < 0.3 ? c.hit : 0) },   // on hits: louder parts only (Outside's quiet intro "flashes a ton")
  alternate: { calm: 0, loud: 1.4, light: (at, r, c, i) => ((i + r + c.no) % 2 ? c.hit : 0.15 * c.hit) },
  strobe: { calm: 0, loud: 0.35, light: (at, r, c) => (c.level > 0.8 ? c.kick : 0.4 * c.kick) },   // loudest parts only (Outside "flashes way too much")
};
function pickPattern(level, bar, last) {
  const names = Object.keys(PATTERNS).filter((n) => n !== last), loudness = Math.min(1, Math.max(0, (level - 0.35) / 0.5));
  const weights = names.map((n) => PATTERNS[n].calm * (1 - loudness) + PATTERNS[n].loud * loudness);
  let x = hash(bar, 7) * weights.reduce((a, w) => a + w, 0);
  for (let i = 0; i < names.length; i++) { x -= weights[i]; if (x <= 0) return names[i]; }
  return names[0];
}

// Big moments and big notes (blasts, the white flash, pyro, stab accents) follow Settings › Big moments (`v.bigMoments`,
// off by default: user, "makes it harder to play right now"); lights.py finds them either way.
// Pyro for big moments: four fountains of sparks shooting up behind the board, falling back and fading (~1.6 s).
const embers = [];
function pyro(ms) {
  if (!show.soft) return;
  for (const [n, x] of [-14, 2, 18, 34].entries()) {
    for (let i = 0; i < 22; i++) {
      let e = embers.find((p) => !p.sp.visible);
      if (!e) {
        e = { sp: new THREE.Sprite(new THREE.SpriteMaterial({ map: show.soft, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, fog: false, toneMapped: false })) };
        scene.add(e.sp); embers.push(e);
      }
      e.born = ms + n * 60 + i * 4; e.x = x + (Math.random() - 0.5) * 1.5;
      e.v = [(Math.random() - 0.5) * 4, 11 + Math.random() * 7, (Math.random() - 0.5) * 3];
      e.sp.material.color.set(i % 3 ? '#ffd27a' : '#ffffff'); e.sp.visible = true;
    }
  }
}
function updateEmbers(ms) {
  for (const e of embers) {
    if (!e.sp.visible) continue;
    const a = (ms - e.born) / 1000;
    if (a > 1.6) { e.sp.visible = false; continue; }
    if (a < 0) { e.sp.material.opacity = 0; continue; }
    e.sp.position.set(e.x + e.v[0] * a, -14 + e.v[1] * a - 7 * a * a, -40 + e.v[2] * a);
    e.sp.material.opacity = 0.9 * (1 - a / 1.6);
    e.sp.scale.setScalar(1.6 * (1 - 0.5 * a / 1.6));
  }
}
// A big moment's white flash over everything (a plane just in front of the camera). The board itself never moves:
// a zoom punch and a shake were tried and dropped (user: "I don't want the board to move at all during big moment,
// just the lighting"; "I don't like the screen shake").
function pop(ms) {
  if (!show) return;
  if (!show.flashPlane) {
    show.flashPlane = new THREE.Mesh(new THREE.PlaneGeometry(10, 10), new THREE.MeshBasicMaterial({ color: '#fff4e0', transparent: true, opacity: 0,
      blending: THREE.AdditiveBlending, depthTest: false, depthWrite: false, fog: false, toneMapped: false }));
    show.flashPlane.position.z = -0.5; show.flashPlane.renderOrder = 999; camera.add(show.flashPlane); scene.add(camera);
  }
  show.flashPlane.material.opacity = 0.2 * show.flash;
  show.flashPlane.visible = show.flash > 0.01;
}

function updateShow(ms, v, pulse) {
  if (!show) buildShow();
  const { beat, down } = beatAt(v.bars, v.bpm, v.t, v.playing);
  // From the song itself when its cues are ready (index.html lightCues): kicks, snare / cymbal hits, loudness, accents
  // and big moments. Until then the tab's beats stand in for the kicks.
  const cue = v.cues;
  const kick = cue ? Math.exp(-cue.kick * 6) : Math.exp(-beat * 6) * (down ? 1 : 0.55), snap = cue ? 0.7 * Math.exp(-cue.snare * 9) : 0;   // softer than they were
  const hit = Math.max(kick, 0.8 * snap), level = cue ? cue.level : 0.6;
  // Quiet parts dim the show, loud ones lift it (user: "when things go quite, dim the lights"): about a quarter in
  // near silence, half in a quiet verse, a little over full in the loudest parts. The board's own light stays.
  const loud = cue ? 0.25 + 1.0 * level ** 1.2 : 1;
  // How much the lights react to hits at all: none in quiet parts (below ~0.35 of the song's loudness), fully from
  // ~0.65 up. Quiet parts keep only slow, dim sweeps (user: "outside still flashes a ton the begining when it is a
  // quiter part": the sparkle pattern and every one of your notes flashed there).
  const drive = cue ? Math.min(1, Math.max(0, (level - 0.35) / 0.3)) : 1;
  // Big moments (user: "make big moments look bigger"): a chorus or drop coming in (from the song), your multiplier
  // going up or a streak milestone (from the game). Everything flares white, grows and the glow behind floods the
  // stage, settling over ~1.5 s. Accents (a big kick and crash together) are a smaller version.
  const bigAge = Math.min(cue ? cue.big : 9, (ms - (v.moment || -1e9)) / 1000);
  const blast = v.bigMoments ? Math.exp(-bigAge * 1.8) * (bigAge < 3 ? 1 : 0) : 0, accent = v.bigMoments && cue ? Math.exp(-cue.accent * 7) * 0.8 : 0;   // accents: big notes (stabs) too
  const boom = Math.max(blast, accent);
  // Real pop in the loud parts (user: "id like for louder moments to have some real pop"): from ~0.55 of the song's
  // loudness up, a kick punches the lighting: lights flare harder, the glow behind and the bloom jump (the board stays put).
  // At most one punch every 0.9 s (on every kick of a loud song it flashed nonstop: user, "outside for example
  // flashes way too much").
  const lift = cue ? Math.min(1, Math.max(0, (level - 0.55) / 0.3)) : 0;
  if (cue && lift > 0.3 && cue.kick < 0.03 && ms - (show.punchAt || -1e9) > 900) show.punchAt = ms;
  const punch = Math.max(lift * Math.exp(-(ms - (show.punchAt || -1e9)) / 1000 * 7), accent);
  show.punch = punch; show.blast = blast; show.flash = v.bigMoments ? Math.exp(-bigAge * 7) * (bigAge < 1 ? 1 : 0) : 0;   // big moments only: on every stab it washed the board grey
  const bigKey = `${cue ? cue.bigNo : -1}|${v.moment || 0}`;
  if (v.bigMoments && bigAge < 0.1 && bigKey !== show.bigKey) { show.bigKey = bigKey; pyro(ms); }   // fountains of sparks at the back
  const m = v.mood, k = m < 0.55 ? m / 0.55 : (m - 0.55) / 0.45, [from, to] = m < 0.55 ? [SHOW_COLORS[0], SHOW_COLORS[1]] : [SHOW_COLORS[1], SHOW_COLORS[2]];
  const ca = from[0].clone().lerp(to[0], k), cb = from[1].clone().lerp(to[1], k);
  const energy = (0.35 + 0.65 * Math.min(1, m / 0.55)) * (0.75 + 0.1 * (v.mult || 1)) * loud;   // brighter with the multiplier
  // A new bar: a new pattern, and the moving ones start again.
  if (v.playing && v.bars?.length) {
    let bar = 0; while (bar + 1 < v.bars.length && v.bars[bar + 1] <= v.t) bar++;
    if (bar !== show.lastBar) { show.lastBar = bar; show.chaseAt = ms; show.pattern = pickPattern(level, bar, show.pattern); }
  }
  const pattern = PATTERNS[show.pattern] || PATTERNS.sweepRight;
  // A held sung note (user: "sync it up with lyrics where they go oh ah"; lights.py's held vocal notes): the back two
  // rigs fill with a warm gold wash spreading out from the middle as the note goes on, and the glow behind turns gold;
  // it fades out over 0.4 s after the note. A new vocal phrase brings a soft glow up from the middle.
  const sg = cue?.sing;
  const sing = sg ? Math.min(1, sg.at / 0.35) * (sg.at > sg.len ? Math.max(0, 1 - (sg.at - sg.len) / 0.4) : 1) * (0.5 + 0.5 * sg.loud) : 0;
  const singReach = sg ? Math.min(1.1, (sg.at / sg.len) * 1.1) : 0, phrase = cue ? Math.exp(-cue.phrase * 4) * 0.4 : 0;
  const now = { go: (ms - show.chaseAt) / (700 + 800 * (1 - drive)), kick: kick * drive, hit: hit * drive, level, no: cue ? Math.max(0, cue.kickNo) + Math.max(0, cue.snareNo) : Math.floor(beat) };
  // Your hits flare a light (front rig) in the string's colour.
  for (const b of v.bursts) {
    if (b.at <= show.lastBurst) continue;
    const row = show.rows[0], l = row.lights[Math.floor(Math.random() * row.lights.length)];
    l.flare = 1; l.flareCol.set(v.colors[b.s % v.colors.length]);
  }
  show.lastBurst = Math.max(show.lastBurst, ...v.bursts.map((b) => b.at));
  show.wall.material.color.copy(ca).lerp(GOLD, 0.5 * sing).lerp(WHITE, 0.45 * boom);
  show.wall.material.opacity = (0.16 + 0.14 * kick * drive) * energy + 0.55 * boom + 0.18 * punch + 0.25 * sing + (v.onFire ? 0.15 * kick * loud : 0);
  show.wall.scale.setScalar(1 + 0.25 * blast);
  if (mats.bloom) mats.bloom.strength = 0.35 + 0.45 * blast + 0.1 * accent + 0.12 * punch;
  for (const row of show.rows) {
    row.rig.rotation.z = Math.sin(ms * 0.00018 + row.r * 2.1) * (0.06 + 0.05 * level);   // rigs swaying, more when loud
    row.rig.position.x = 14 + Math.sin(ms * 0.00011 + row.r) * 6;
    row.lights.forEach((l, i) => {
      const at = i / Math.max(1, row.lights.length - 1), twinkle = 0.75 + 0.25 * Math.sin(ms * 0.002 + l.phase);
      const lit = pattern.light(at, row.r, now, i);
      let o = (0.08 + 0.08 * drive * hit * (row.r === 0 ? kick : snap) + (0.2 + 0.18 * drive) * lit * (1 + 0.4 * lift)) * energy * twinkle + 0.45 * boom + 0.12 * punch * lit;
      const col = (i + row.r) % 2 ? cb : ca;
      l.glow.material.color.copy(col);
      if (l.flare > 0.01) { l.glow.material.color.lerp(l.flareCol, l.flare * (0.4 + 0.6 * drive)); o += 0.22 * l.flare * (0.25 + 0.75 * drive); l.flare *= 0.9; }   // soft: bright ones bloomed over the board
      if (row.r > 0 && (sing > 0.01 || phrase > 0.01)) {   // the vocal wash on the back rigs
        const spread = Math.abs(at - 0.5) * 2, wash = sing * (spread <= singReach ? 1 : Math.max(0, 1 - (spread - singReach) * 6)) + phrase * Math.max(0, 1 - spread * 2);
        l.glow.material.color.lerp(GOLD, Math.min(1, 0.7 * wash)); o += 0.35 * wash; l.washed = wash;
      } else l.washed = 0;
      if (boom > 0.02) l.glow.material.color.lerp(WHITE, 0.55 * boom);
      if (v.onFire && accent > 0.4) { l.glow.material.color.lerp(WHITE, 0.4); o += 0.12 * loud; }   // x4: white on the accents (every kick was too much)
      l.glow.material.opacity = Math.min(0.5 + 0.4 * boom, o);
      l.glow.scale.setScalar((l.size ||= l.glow.scale.x) * (1 + 0.3 * lit * (0.4 + 0.6 * drive) + 0.9 * boom + 0.35 * punch + 0.5 * l.washed));
    });
  }
}

const laneZ = (s) => ((lanes.n - 1) / 2 - s) * LANE;   // low string nearest the camera = bottom of the screen

// Strings, colour bands, pads and names for n strings (4 bass, 5 five-string, 6 guitar).
// Looks being tried on the board lab page (static/board-lab.html; user: "can we make the board look cooler?", picked
// glass reflections, living strings and new strike pads, "lets do mockups"), each switched by `v.look`; off, the
// board is as before.
// - reflect: a soft coloured reflection on the glass under each note and in front of each pad, like a polished floor
//   (sprites on the surface: a real mirror pass would draw the scene twice, against the 120 fps aim).
// - strings: real strings (user, after the mockups: "the strings are pretty distracting though. Can we make them look
//   like real strings?"; ripples running along them and beat pulses were tried): thicker and graded like a bass set,
//   bright round-wound nickel with fine ridges and only a hint of the string's colour, shimmering for a moment near
//   the pad when a note is played. The new pads (a glowing ring) were tried too; the user kept the old ones.
// - minimal (user: "the strings look a little weird. Would a minimalist design work better?"): strings as thin flat
//   matte lines in their colour (no tube, glow or metal), brightening for a moment when played; a plain matte board
//   with no light edges and the lane colours barely there; flat matte notes. The old pads stay.
// Highway looks (user: "can we improve the look of the highway?"):
// - depth: the glass lit near the pads and falling away into the dark, each string's lane brightest at the pads, and a
//   groove with a faint lit edge between the lanes, so the board reads as a surface rather than a flat slab.
// - strike: a glowing strike line through the pads (brighter on hits, blue at x4), faint beat lines between the bar
//   lines, and every bar / beat line lighting up as it crosses the strike line, so the rhythm shows where you play.
// - gems: notes in their string's full colour with a soft glow of their own and a lit underside (the dark skirt
//   made them look like boxes on the board).
let softTex = null, streakTex = null, windTex = null, depthTex = null, bandTex = null, strikeTex = null;
// A texture along the board (80 long, x -10 to 70): stops as [x, colour].
function alongBoard(stops) {
  const c = document.createElement('canvas'); c.width = 1024; c.height = 4; const g = c.getContext('2d');
  const grad = g.createLinearGradient(0, 0, 1024, 0);
  stops.forEach(([x, col]) => grad.addColorStop((x + 10) / 80, col));
  g.fillStyle = grad; g.fillRect(0, 0, 1024, 4);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; t.anisotropy = 8; return t;
}
function looksTextures() {
  if (softTex) return;
  depthTex = alongBoard([[-10, '#07080c'], [-1, '#11141d'], [0.5, '#191d29'], [8, '#0e1017'], [25, '#08090d'], [70, '#050507']]);
  bandTex = alongBoard([[-10, 'rgba(255,255,255,0.3)'], [0, 'rgba(255,255,255,1)'], [6, 'rgba(255,255,255,0.55)'], [20, 'rgba(255,255,255,0.2)'], [70, 'rgba(255,255,255,0.05)']]);
  const sc = document.createElement('canvas'); sc.width = 128; sc.height = 4; const sg = sc.getContext('2d');   // soft across, along x
  const sgr = sg.createLinearGradient(0, 0, 128, 0);
  sgr.addColorStop(0, 'rgba(255,255,255,0)'); sgr.addColorStop(0.5, 'rgba(255,255,255,1)'); sgr.addColorStop(1, 'rgba(255,255,255,0)');
  sg.fillStyle = sgr; sg.fillRect(0, 0, 128, 4); strikeTex = new THREE.CanvasTexture(sc);
  softTex = gradientTexture([[0, 'rgba(255,255,255,1)'], [0.4, 'rgba(255,255,255,0.45)'], [1, 'rgba(255,255,255,0)']], false);
  const c = document.createElement('canvas'); c.width = 64; c.height = 256; const g = c.getContext('2d');
  const grad = g.createLinearGradient(0, 0, 0, 256);                // a reflection: brightest at the note, fading away
  grad.addColorStop(0, 'rgba(255,255,255,0.9)'); grad.addColorStop(0.35, 'rgba(255,255,255,0.35)'); grad.addColorStop(1, 'rgba(255,255,255,0)');
  g.fillStyle = grad; g.fillRect(0, 0, 64, 256);
  const side = g.createLinearGradient(0, 0, 64, 0);
  side.addColorStop(0, 'rgba(0,0,0,1)'); side.addColorStop(0.25, 'rgba(0,0,0,0)'); side.addColorStop(0.75, 'rgba(0,0,0,0)'); side.addColorStop(1, 'rgba(0,0,0,1)');
  g.globalCompositeOperation = 'destination-out'; g.fillStyle = side; g.fillRect(0, 0, 64, 256);
  streakTex = new THREE.CanvasTexture(c);
  // Round-wound: a ridge for each turn of the wrap wire, as a bump map repeated along the string.
  const w = document.createElement('canvas'); w.width = 4; w.height = 32; const wg = w.getContext('2d');
  const ridge = wg.createLinearGradient(0, 0, 0, 32);
  ridge.addColorStop(0, '#000'); ridge.addColorStop(0.5, '#fff'); ridge.addColorStop(1, '#000');
  wg.fillStyle = ridge; wg.fillRect(0, 0, 4, 32);
  windTex = new THREE.CanvasTexture(w); windTex.wrapS = windTex.wrapT = THREE.RepeatWrapping; windTex.repeat.set(1, 700); windTex.anisotropy = 8;
}
// A string that can shimmer: after a note is played (uAge seconds ago, uAmp how far), it moves fast and slightly near
// the pad, settling over well under a second, as a plucked string looks.
function realString(material) {
  const u = { uAmp: { value: 0 }, uAge: { value: 0 } };
  material.userData.u = u;
  material.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, u);
    shader.vertexShader = 'uniform float uAmp; uniform float uAge;\n' + shader.vertexShader.replace('#include <begin_vertex>', `#include <begin_vertex>
      float xw = 30.0 - position.y;                                   // along the board (the string lies turned a quarter)
      transformed.x += uAmp * sin(uAge * 95.0 + xw * 0.35) * exp(-max(xw, 0.0) * 0.08);`);
  };
  material.customProgramCacheKey = () => 'real-string';
  return material;
}

function buildLanes(n, colors, names) {
  looksTextures();
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
    const str = add(new THREE.Mesh(new THREE.CylinderGeometry(r, r, 80, 12, 160), realString(new THREE.MeshStandardMaterial({ color: '#8a847c', metalness: 0.6, roughness: 0.5, emissive: col, emissiveIntensity: 0.18 })))); str.rotation.z = Math.PI / 2; str.position.set(30, 0.05, laneZ(s)); (lanes.strings ||= []).push(str);
    str.userData.r = r; str.userData.col = col;
    const line = add(new THREE.Mesh(new THREE.PlaneGeometry(80, 0.04 - 0.012 * (n > 1 ? s / (n - 1) : 0)), new THREE.MeshBasicMaterial({ color: new THREE.Color(col).multiplyScalar(0.8) })));
    line.rotation.x = -Math.PI / 2; line.position.set(30, 0.012, laneZ(s)); line.visible = false; (lanes.lines ||= []).push(line);   // minimal
    const rim = add(new THREE.Mesh(new RoundedBoxGeometry(NOTE_W + 0.16, 0.1, 0.9, 4, 0.08), new THREE.MeshStandardMaterial({ color: '#000', emissive: col, emissiveIntensity: 0.55 })));
    rim.position.set(0, 0.06, laneZ(s));
    const lens = add(new THREE.Mesh(new RoundedBoxGeometry(NOTE_W + 0.04, 0.12, 0.78, 4, 0.08), new THREE.MeshPhysicalMaterial({ color: '#100c0a', roughness: 0.2, clearcoat: 1, emissive: col, emissiveIntensity: 0.12 })));
    lens.position.set(0, 0.13, laneZ(s));
    const light = add(new THREE.PointLight(col, 0, 4, 2)); light.position.set(0.3, 0.8, laneZ(s));
    const name = add(new THREE.Sprite(new THREE.SpriteMaterial({ map: label(names[s] || '', col, true, 128, true), depthWrite: false })));
    name.scale.set(0.62, 0.62, 1); name.position.set(NUT - 0.5, 0.35, laneZ(s));
    name.visible = false;                                // drawn crisply on the HUD instead (nameScreen); kept for the camera fit
    const flat = (m, y) => { m.rotation.x = -Math.PI / 2; m.position.set(0, y, laneZ(s)); m.visible = false; return add(m); };
    // The pad's reflection on the glass (look.reflect): a pool of its light in front of it.
    const pool = flat(new THREE.Mesh(new THREE.PlaneGeometry(1.5, 1.4), new THREE.MeshBasicMaterial({ map: softTex, color: col, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false })), 0.006);
    pool.position.z = laneZ(s) + 0.35;
    lanes.pads.push({ rim, lens, light, col, hitAt: -1e9, pool });
    lanes.names.push(name);
  }
  lanes.boardW = boardW;
  lanes.edges = [-1, 1].map((side) => {                  // glass: a thin light edge along each side
    const e = add(new THREE.Mesh(new THREE.BoxGeometry(80, 0.03, 0.03), new THREE.MeshBasicMaterial({ color: '#fff5e6', transparent: true, opacity: 0.35, toneMapped: false })));
    e.position.set(30, 0.01, side * (boardW / 2)); return e;
  });
  // look.depth: a groove between lanes, with a faint lit edge on its near side.
  lanes.grooves = [];
  for (let s = 0; s < n - 1; s++) {
    const z = laneZ(s) - LANE / 2;
    const groove = add(new THREE.Mesh(new THREE.PlaneGeometry(80, 0.07), new THREE.MeshBasicMaterial({ color: '#000', transparent: true, opacity: 0.55, depthWrite: false })));
    groove.rotation.x = -Math.PI / 2; groove.position.set(30, 0.003, z);
    const lit = add(new THREE.Mesh(new THREE.PlaneGeometry(80, 0.014), new THREE.MeshBasicMaterial({ color: '#fff5e6', transparent: true, opacity: 0.08, depthWrite: false, toneMapped: false })));
    lit.rotation.x = -Math.PI / 2; lit.position.set(30, 0.004, z + 0.045);
    groove.visible = lit.visible = false; lanes.grooves.push(groove, lit);
  }
  // look.strike: the strike line through the pads and its soft glow.
  lanes.strike = add(new THREE.Mesh(new THREE.PlaneGeometry(0.06, boardW), new THREE.MeshBasicMaterial({ color: '#fff1d6', transparent: true, opacity: 0.6, depthWrite: false, toneMapped: false })));
  lanes.strike.rotation.x = -Math.PI / 2; lanes.strike.position.set(0, 0.005, 0); lanes.strike.visible = false;
  lanes.strikeGlow = add(new THREE.Mesh(new THREE.PlaneGeometry(1.4, boardW), new THREE.MeshBasicMaterial({ map: strikeTex, color: '#ffc46b', transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false })));
  lanes.strikeGlow.rotation.x = -Math.PI / 2; lanes.strikeGlow.position.set(0, 0.006, 0); lanes.strikeGlow.visible = false;
  lanes.beats = [];
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
  const line = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.014, lanes.boardW), mats.line.clone()); line.position.y = 0.008;   // its own: lights up crossing the strike line (look.strike)
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
  const cap = new THREE.Mesh(mats.capGeo, new THREE.MeshStandardMaterial({ transparent: true, roughness: 0.4 }));
  cap.visible = false;                                   // where a long note ends: let go there
  const refl = new THREE.Mesh(mats.reflGeo, new THREE.MeshBasicMaterial({ map: streakTex, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false, toneMapped: false }));
  refl.rotation.x = -Math.PI / 2; refl.visible = false;  // its reflection on the glass (look.reflect)
  grp.add(base, body, num); world.add(grp); world.add(tail); world.add(tailGlow); world.add(cap); world.add(refl);
  return { grp, body, base, num, tail, tailGlow, cap, refl, k: -1, sinkAt: 0 };
}

function setFretboard(style) {
  if (lanes.style === style) return;
  lanes.style = style;
  Object.assign(board.material, mats.boards[style] || mats.boards.glass); board.material.needsUpdate = true;
  lanes.bands.forEach((b, s) => { b.material.opacity = style === 'glass' ? (s % 2 ? 0.035 : 0.022) : style === 'dark' ? (s % 2 ? 0.07 : 0.045) : 0.1; });
  lanes.strGlow = style === 'glass' ? 0.6 : 0.18;      // glass: the strings are thin glowing lines
  lanes.edges?.forEach((e) => { e.visible = style === 'glass'; });
  if (lanes.minimal) {                                   // minimal: plain matte board, lanes barely there, no edges
    Object.assign(board.material, { map: null, color: new THREE.Color('#0c0c10'), roughness: 1, metalness: 0 }); board.material.needsUpdate = true;
    lanes.bands.forEach((b, s) => { b.material.opacity = s % 2 ? 0.03 : 0.012; });
    lanes.edges?.forEach((e) => { e.visible = false; });
  }
  const depth = lanes.depth && style === 'glass' && !lanes.minimal;
  if (depth) {                                           // depth: lit near the pads, falling away into the dark
    Object.assign(board.material, { map: depthTex, color: new THREE.Color('#ffffff'), roughness: 0.6, metalness: 0.05 }); board.material.needsUpdate = true;
  }
  lanes.bands.forEach((b, s) => {
    const map = depth ? bandTex : null;
    if (b.material.map !== map) { b.material.map = map; b.material.needsUpdate = true; }
    if (depth) b.material.opacity = s % 2 ? 0.075 : 0.055;
  });
  lanes.grooves?.forEach((g) => { g.visible = depth; });
  if (lanes.flow && !lanes.minimal) {                    // flow: facing the stage lights head on, the glass washed out grey
    Object.assign(board.material, { color: board.material.map ? new THREE.Color('#5a5a66') : new THREE.Color('#040508'), roughness: 0.8 }); board.material.needsUpdate = true;
  }
}

// The camera: in front, turned 5 degrees right ("don't make it so angled"), slid sideways until every
// string name sits inside the left edge, measured on screen (a wide window pushed them off).
function aim(w, h) {
  const aspect = w / h, z = 16, yaw = lanes.flow ? 0 : (5 * Math.PI) / 180;   // flow: straight on, so the speed is the same everywhere
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
  pace.from = performance.now() + 3000;
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
// The fret numbers and hand-move labels for the HUD to draw (v.hudNumbers): screen positions (CSS px), sizes, fades.
let hud = { notes: [], moves: [] };
export function labels() { return hud; }
const toScreen = (x, y, z) => { v3.set(x, y, z).applyMatrix4(world.matrixWorld).project(camera); return { x: ((v3.x + 1) / 2) * wrapEl.clientWidth, y: ((1 - v3.y) / 2) * wrapEl.clientHeight }; };

export function padScreen(s) {
  const p = lanes.pads?.[s];                             // none until the first frame built the strings
  if (!p || !wrapEl) return null;
  p.lens.getWorldPosition(v3).project(camera);
  return { x: ((v3.x + 1) / 2) * wrapEl.clientWidth, y: ((1 - v3.y) / 2) * wrapEl.clientHeight };
}

// One frame. v: { t, notes, n, colors, names, Z (seconds ahead), quickGap, prev, result, laneHitMs,
// laneMissMs, sustain, bursts, onFire, mood (0-1), fretboard, bars, bpm, inputLane, legato(k) }
// Sharpness that follows the frame rate, aiming for the screen's own refresh rate (user: "can we make the game run
// at 120fps?"; a MacBook Pro's screen refreshes at 120 Hz). The screen's frame time is measured once while the page
// is idle (the menus): during a game a frame that takes a little over 8.3 ms waits for the next refresh, so a
// 120 Hz screen shows a steady 60 and looks like a 60 Hz one. Full sharpness (the screen's pixel ratio) while frames
// keep up; when they average 15% slower than the screen over 2 s it steps down by a quarter (not below 1). If a step
// down didn't make frames at least 10% quicker it goes back and stays at that sharpness (no blur for nothing). It
// steps back up while frames keep the screen's pace, but never to a sharpness that proved too slow. The first 3 s
// after the board appears (loading), pauses and hidden tabs (gaps over 0.1 s) don't count.
const PR_MAX = Math.min(3, devicePixelRatio);
// Each step resizes the drawing buffers, a ~40 ms hitch, and a song began with up to four of them (2 -> 1.25:
// user, "getting some choppyness again"). So the sharpness that settled is kept on this computer and the next game
// starts there; it still steps if that turns out too slow or there's room.
const PR_KEY = 'encore-pixel-ratio';
const prStart = (() => { try { const v = +localStorage.getItem(PR_KEY); return v >= 1 && v <= PR_MAX ? v : PR_MAX; } catch { return PR_MAX; } })();
const pace = { pr: prStart, slow: Infinity, floor: 1, down: null, sum: 0, n: 0, last: 0, from: 0, screen: 1000 / 60 };
(function measureScreen(left = 90, dts = [], prev = 0) {
  requestAnimationFrame((ms) => {
    if (prev && ms - prev < 100) dts.push(ms - prev);
    if (left > 1) { measureScreen(left - 1, dts, ms); return; }
    dts.sort((a, b) => a - b);
    if (dts.length > 20) pace.screen = Math.max(1000 / 144, Math.min(1000 / 60, dts[dts.length >> 1]));
  });
})();
function keepPace(ms) {
  const dt = ms - pace.last;
  pace.last = ms;
  if (dt > 100 || ms < pace.from) return;
  pace.sum += dt; pace.n++;
  if (pace.sum < 2000) return;
  const avg = pace.sum / pace.n;
  pace.sum = pace.n = 0;
  let pr = pace.pr;
  if (avg > pace.screen * 1.15) {
    // Frames come whole refreshes apart, so one step may not show a gain yet: keep stepping, measured against where
    // the steps began, and only at the bottom with no gain go back there.
    if (pace.down && avg < pace.down.avg * 0.9) pace.down = { pr: pace.pr, avg };   // quicker: a new starting point
    if (pace.pr - 0.25 >= pace.floor) { pace.slow = Math.min(pace.slow, pace.pr); pace.down ||= { pr: pace.pr, avg }; pr = pace.pr - 0.25; }
    else if (pace.down && pace.down.pr !== pace.pr) { pr = pace.down.pr; pace.floor = pr; pace.down = null; }   // no help: back
  } else {
    pace.down = null;
    const up = pace.pr + 0.25;
    if (avg < pace.screen * 1.06 && up <= PR_MAX && up < pace.slow) pr = up;
  }
  if (pr === pace.pr) return;
  pace.pr = pr;
  try { localStorage.setItem(PR_KEY, String(pr)); } catch { /* not kept */ }
  renderer.setPixelRatio(pr); composer.setPixelRatio(pr);
  lastSize = '';                                         // re-sized at the new ratio below
}

export function frame(v) {
  const w = wrapEl.clientWidth, h = wrapEl.clientHeight, ms = performance.now();
  if (!w || !h) return;
  keepPace(ms);
  if (lanes.n !== v.n || lanes.key !== v.names.join()) { buildLanes(v.n, v.colors, v.names); lanes.key = v.names.join(); lastSize = ''; }
  const minimal = !!v.look?.minimal;
  if (lanes.minimal !== minimal) { lanes.minimal = minimal; lanes.style = null; }   // board surface: re-applied below
  const depthLook = !!v.look?.depth;
  if (lanes.depth !== depthLook) { lanes.depth = depthLook; lanes.style = null; }
  // look.flow (user, after mockups: the tab must keep the eyes in one place, so the pads stay put and the tab moves;
  // the leaning board makes notes grow and speed up as they come in): the board turned almost to face the camera, so
  // notes glide at one steady speed, the notes shown as their fret numbers.
  const flowLook = !!v.look?.flow;
  if (lanes.flow !== flowLook) { lanes.flow = flowLook; world.rotation.x = flowLook ? 1.42 : Math.PI / 4; lastSize = ''; lanes.style = null; }
  setFretboard(v.fretboard);
  if (lastSize !== `${w}x${h}`) {
    lastSize = `${w}x${h}`;
    renderer.setSize(w, h, false); composer.setSize(w, h);
    renderer.domElement.style.width = `${w}px`; renderer.domElement.style.height = `${h}px`;
    aim(w, h);
  }
  if (warmFor !== v.notes) prewarm(v);

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
  lanes.wash.material.opacity = lanes.fire * (0.55 + 0.1 * pulse) * (lanes.flow ? 0.4 : 1);   // flow: seen head on, the full wash floods the board
  lanes.bands.forEach((b) => b.material.color.lerpColors(b.userData.c ||= new THREE.Color(b.userData.col), BLUE, lanes.fire));

  updateShow(ms, v, pulse);
  updateEmbers(ms);
  pop(ms);

  // Bar lines.
  const strikeOn = !!v.look?.strike, visibleBars = [];
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
    b.line.material.opacity = 0.14 + (strikeOn ? 0.6 * Math.exp(-Math.abs(x) / 0.35) : 0);   // lights up crossing the strike line
  });
  // look.strike: the strike line, and faint beat lines between the bar lines (four to a bar) lighting up as they cross it.
  lanes.strike.visible = lanes.strikeGlow.visible = strikeOn;
  const beatXs = [];
  if (strikeOn) {
    const fireCol = v.onFire ? '#7dd3fc' : '#ffc46b';
    lanes.strike.material.color.set(v.onFire ? '#e0f2fe' : '#fff1d6');
    lanes.strikeGlow.material.color.set(fireCol);
    lanes.strikeGlow.material.opacity = 0.14 + 0.3 * pulse;
    for (let i = 0; i + 1 < v.bars.length; i++) {
      const b0 = v.bars[i], step = (v.bars[i + 1] - b0) / 4;
      if ((b0 - t) * speed > far + 1) break;
      for (let j = 1; j < 4; j++) { const x = (b0 + j * step - t) * speed; if (x > NUT - 0.2 && x < far) beatXs.push(x); }
    }
  }
  while (lanes.beats.length < beatXs.length) {
    const m = new THREE.Mesh(new THREE.BoxGeometry(0.03, 0.01, lanes.boardW), new THREE.MeshBasicMaterial({ color: '#ffe9c9', transparent: true, opacity: 0.05, depthWrite: false }));
    m.position.y = 0.007; world.add(m); lanes.objects.push(m); lanes.beats.push(m);
  }
  lanes.beats.forEach((m, j) => {
    m.visible = j < beatXs.length;
    if (!m.visible) return;
    const x = beatXs[j]; m.position.x = x;
    m.material.opacity = (0.05 + 0.4 * Math.exp(-Math.abs(x) / 0.3)) * Math.max(0, Math.min(1, (far - x) / FADE));
  });

  // Hand moves: a dashed amber line across the board just before the hand moves along the neck.
  const movesAll = (v.moves || []).map((m, j) => [(m - t) * speed, v.moveTargets?.[j]]).filter(([x]) => x > 0.3 && x < far);
  const moves = movesAll.map(([x]) => x);
  hud = { notes: [], moves: [] };
  if (v.hudNumbers) for (const [x, to] of movesAll) {
    if (!to) continue;
    const p = toScreen(x, 0.05, laneZ(lanes.n - 1) - LANE * 0.5);   // above the top string
    hud.moves.push({ x: p.x, y: p.y, fret: to[0], up: to[1], alpha: 0.95 * Math.max(0, Math.min(1, (far - x) / FADE)) });
  }
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
  for (const [k, o] of used) if (!live.has(k)) { o.grp.visible = false; o.tail.visible = false; o.tailGlow.visible = false; o.cap.visible = false; o.refl.visible = false; o.k = -1; pool.push(o); used.delete(k); }
  const held = new Set(), heldEnds = new Set();         // strings with a note held now / ending now
  const look = v.look || {}, coming = new Array(lanes.n).fill(0);   // coming: how close each string's next note is (0-1)
  for (const k of live) {
    const [ns, ne, s, f] = notes[k];
    if (s >= lanes.n) continue;
    let o = used.get(k);
    const res = v.result[k], missed = res === 'miss', open = f === 0, dropped = !!v.hold?.[k]?.lost;   // a hold you let go of
    // The first note has its number; the same fret again on that string is the thin bar (user: "I wanted the first
    // note to have a number and any repeating to have the thinner bar"); after a rest of 2.5 s it's a full note again.
    // A repeat is the very next note played, same string and fret, nothing in between (A5 E0 A5 isn't one: the user
    // read those as new notes, "there's not repeating notes that are not showing").
    let pk = k - 1;
    while (pk >= 0 && ns - notes[pk][0] < 0.03 && notes[pk][2] !== s) pk--;   // skip notes struck together with it
    const tick = pk >= 0 && notes[pk][2] === s && notes[pk][3] === f && ns - notes[pk][0] < 2.5, repeat = false;
    // Once the numbered note has reached the pads, the next tick carries the number (user: Flower Power Madness was
    // hard to read in the game: a run of repeats left bare ticks on the board with no fret anywhere).
    const lead = tick && (notes[pk][0] - t) * speed < NUT + 0.6;
    if (!o) {
      o = pool.pop() || noteObject(); used.set(k, o); o.k = k; o.sinkAt = 0; o.style = '';
    }
    const gems = !!look.gems && !look.minimal;
    const style = `${s}|${f}|${missed}|${tick}|${repeat}|${v.onFire}|${lead}|${dropped}|${gems}|${!!look.minimal}`;
    if (o.style !== style) {                            // (re)paint for this note
      o.style = style;
      const col = new THREE.Color(v.colors[s % v.colors.length]);
      o.body.material.color.copy(missed ? new THREE.Color('#3f3f46') : open ? new THREE.Color('#120e0b') : look.minimal ? col.clone().multiplyScalar(0.8) : gems ? col.clone().multiplyScalar(0.9) : lanes.paint[s]);
      Object.assign(o.body.material, look.minimal ? { roughness: 0.9, clearcoat: 0 } : gems ? { roughness: 0.25, clearcoat: 1 } : { roughness: 0.4, clearcoat: 0.6 });   // minimal: flat matte
      o.body.material.emissive.copy(v.onFire && !missed ? new THREE.Color('#38bdf8') : col); o.body.material.emissiveIntensity = missed ? 0 : v.onFire ? 0.08 : open ? 0.06 : 0;
      o.base.material.color.copy(missed ? new THREE.Color('#18181b') : open || gems ? col : new THREE.Color('#0a0705'));
      o.base.material.emissive.copy(v.onFire && gems && !missed ? new THREE.Color('#38bdf8') : col);
      o.base.material.emissiveIntensity = missed ? 0 : open ? 0.8 : gems ? 0.9 : 0;   // gems: a lit underside
      const key = `${f}|${open}|${s}|${missed}`;
      tex[key] ||= label(String(f), missed ? '#a1a1aa' : open ? v.colors[s % v.colors.length] : '#ffffff', !open && !missed);
      o.num.material.map = tex[key]; o.num.material.needsUpdate = true;
      o.num.visible = (!tick || lead) && !repeat;
      o.numShown = o.num.visible;
      const wide = tick ? 0.14 / 0.86 : NOTE_W / 0.86;
      o.body.scale.x = o.base.scale.x = wide;
      o.tail.material.color.copy(missed || dropped ? new THREE.Color('#3f3f46') : col);
      o.cap.material.color.copy(missed || dropped ? new THREE.Color('#3f3f46') : col); o.cap.material.emissive.copy(col);
    }
    // Where it is: hit notes sink into the pad (90 ms); held long notes sit on the pad.
    let end = Math.max(ne, ns);
    for (let j = k + 1; j < notes.length && notes[j][0] < end; j++) if (notes[j][2] === s) { end = notes[j][0]; break; }
    const holding = res === 'hit' && ns <= t && t < end && end - ns > 0.15 && !dropped;
    if (res === 'hit' && !o.sinkAt) o.sinkAt = ms;
    const sink = res === 'hit' && !(holding || (dropped && t < end)) ? Math.min(1, (ms - o.sinkAt) / 90) : 0;
    const x = res === 'hit' ? 0 : (ns - t) * speed;
    const come = Math.max(0, Math.min(1, (far - x) / FADE)), ease = come * come * (3 - 2 * come);   // fading in from the distance
    o.grp.visible = (res !== 'hit' || holding || sink < 1) && !(dropped && res === 'hit') && x > NUT + 0.2 && x < far;
    o.grp.position.set(x, (look.flow ? 0 : 0.35 * (1 - ease)) - 0.25 * sink, laneZ(s));
    o.grp.scale.setScalar((1 - 0.5 * sink) * (look.flow ? 1 : 0.85 + 0.15 * ease));
    if (look.flow !== o.flow) {                         // flow: just the number (a quick repeat keeps its slim tick)
      o.flow = look.flow;
      o.num.scale.setScalar(look.flow ? 1.7 : 1);
      o.num.position.y = look.flow ? 0.12 : 0.335;
    }
    if (look.flow) o.body.visible = o.base.visible = tick && !lead;
    else if (!o.body.visible) o.body.visible = o.base.visible = true;
    const fade = ease * (missed ? 0.7 : 1);
    o.body.material.opacity = o.base.material.opacity = fade; o.num.material.opacity = fade;
    if (v.hudNumbers) {                                  // upright on the HUD instead of flat on the note
      o.num.visible = false;
      if (o.numShown && o.grp.visible && res !== 'hit') {
        const a = toScreen(x, 0.42, laneZ(s)), b = toScreen(x + NOTE_W, 0.42, laneZ(s));
        hud.notes.push({ x: a.x, y: a.y, s, text: String(f), open, missed, alpha: Math.min(1, fade * 1.15), size: Math.round(Math.max(17, Math.min(34, Math.abs(b.x - a.x) * 1.05))) });
      }
    }
    if (!res && ns >= t && ns - t < 0.6) coming[s] = Math.max(coming[s], 1 - (ns - t) / 0.6);
    o.refl.visible = !!look.reflect && o.grp.visible && !missed;
    if (o.refl.visible) {                                // its reflection on the glass, toward the near edge
      const w = tick && !lead ? 0.3 : NOTE_W * 1.6;
      o.refl.scale.set(w, 1.1, 1); o.refl.position.set(x, 0.008, laneZ(s) + 0.5 + 0.1 * (1 - sink));
      o.refl.material.color.copy(v.onFire ? BLUE : (lanes.pads[s].c ||= new THREE.Color(v.colors[s % v.colors.length])));
      o.refl.material.opacity = 0.32 * fade * (1 - sink);
    }
    // A tail as long as the note rings, wider on long notes so they read as holds, with a cap where it ends
    // (user: "can we improve how held notes look? Also, I always lose it at the end"). A dropped hold's tail goes grey.
    const x0 = Math.max(0, x), x1 = Math.min(far, (end - t) * speed), long = end - ns > 0.4;
    o.tail.visible = end - ns > 0.15 && x1 > x0 + 0.1 && (o.grp.visible || (dropped && t < end));
    o.tail.scale.z = holding ? 2.4 : long ? 1.7 : 1;
    const capX = (end - t) * speed;
    o.cap.visible = long && o.tail.visible && capX < far && capX > x0 + 0.15;
    if (o.cap.visible) {
      o.cap.position.set(capX, 0.08, laneZ(s));
      const near = holding ? Math.max(0, 1 - (end - t) / 0.4) : 0;   // lights up as you reach it
      o.cap.material.opacity = dropped ? 0.4 : fade; o.cap.material.emissiveIntensity = holding ? 0.6 + 1.4 * near : 0.15;
      o.cap.scale.setScalar(1 + 0.35 * near);
    }
    // Holding (user: "make the effect of holding a note look better"): the tail burns bright with a soft glow
    // round it and drains into the pad, the note glows on the pad, pulsing.
    const throb = 0.5 + 0.5 * Math.sin(ms / 70);
    if (o.tail.visible) {
      o.tail.scale.x = x1 - x0; o.tail.position.set((x0 + x1) / 2, 0.06, laneZ(s));
      o.tail.material.opacity = holding ? 0.95 : dropped ? 0.35 : 0.55 * fade; o.tail.material.emissive.copy(o.tail.material.color);
      o.tail.material.emissiveIntensity = holding ? 1.1 + 0.5 * throb : 0;
    }
    o.tailGlow.visible = o.tail.visible && holding;
    if (o.tailGlow.visible) {
      o.tailGlow.scale.set(x1 - x0, 1.6, 3.2); o.tailGlow.position.set((x0 + x1) / 2, 0.07, laneZ(s));
      o.tailGlow.material.color.copy(v.onFire ? BLUE : lanes.pads[s].c ||= new THREE.Color(v.colors[s % v.colors.length]));
      o.tailGlow.material.opacity = 0.28 + 0.14 * throb;
    }
    if (holding) { o.body.material.emissiveIntensity = 0.45 + 0.3 * throb; held.add(s); if (end - t < 0.06) heldEnds.add(s); }
    else if (o.style.endsWith('false') && !missed && !open) o.body.material.emissiveIntensity = gems ? 0.22 : 0;   // gems: a soft glow of their own
  }

  // Pads: a hit dips the pad and lights its lens and a pool of light; at x4 everything goes blue;
  // what you're playing now outlines its pad in white.
  for (const [s, p] of lanes.pads.entries()) {
    const at = v.laneHitMs[s] ?? -1e9, age = ms - at, k0 = Math.max(0, 1 - age / 220), hitDip = age < 60 ? age / 60 : Math.max(0, 1 - (age - 60) / 120);
    // The pluck heard (laneTapMs, before the hit is confirmed): the pad dips and lights at once, a little less than a hit.
    const tapAge = ms - (v.laneTapMs?.[s] ?? -1e9), press = tapAge < 160 ? 1 - tapAge / 160 : 0;
    const k = Math.max(k0, 0.6 * press), dip = Math.max(hitDip, 0.85 * press);
    const bad = Math.max(0, 1 - (ms - (v.laneMissMs[s] ?? -1e9)) / 300), held = ms - (v.sustain[s] ?? -1e9) < 120;
    const col = v.onFire ? '#38bdf8' : p.col;
    p.lens.material.emissive.set(bad > k ? '#ef4444' : col); p.lens.material.emissiveIntensity = 0.12 + 1.5 * Math.max(k, bad * 0.6, held ? 0.6 + 0.3 * Math.sin(ms / 70) : 0);
    p.light.color.set(col); p.light.intensity = 4 * Math.max(k, held ? 0.5 : 0);
    p.lens.position.y = 0.13 - 0.06 * dip; p.rim.position.y = 0.06 - 0.04 * dip;
    p.rim.material.emissive.set(v.inputLane === s ? '#ffffff' : bad > 0.3 ? '#ef4444' : col);
    p.rim.material.emissiveIntensity = v.inputLane === s ? 1.1 : 0.55;
    p.pool.visible = !!look.reflect;
    if (p.pool.visible) { p.pool.material.color.set(bad > k ? '#ef4444' : col); p.pool.material.opacity = 0.1 + 0.45 * Math.max(k, 0.5 * coming[s], held ? 0.5 : 0); }
  }
  // Held strings vibrate and glow; sparks stream off the pad; a small burst when the hold reaches its end.
  // Real strings (look.strings): graded thickness, wound nickel, a short shimmer when a note is played.
  lanes.strings?.forEach((str, s) => {
    const line = lanes.lines?.[s];
    str.visible = !look.minimal;
    if (line) {
      line.visible = !!look.minimal;
      if (line.visible) {                                // a moment brighter when played, steady while held
        const flash = Math.max(Math.exp(-(ms - (v.laneHitMs[s] ?? -1e9)) / 250), held.has(s) ? 0.5 : 0);
        line.material.color.set(v.onFire ? '#38bdf8' : lanes.pads[s].col).multiplyScalar(0.8).lerp(WHITE, 0.5 * flash);
      }
    }
    const real = !!look.strings && !look.minimal;
    if (str.userData.real !== real) {                    // switch the string's surface
      str.userData.real = real;
      const m = str.material;
      if (real) {
        const k = lanes.n > 1 ? s / (lanes.n - 1) : 0, thick = 0.075 - 0.035 * k;   // low string ~2x the high one, like a bass set
        str.scale.set(thick / str.userData.r, 1, thick / str.userData.r);
        Object.assign(m, { color: new THREE.Color('#9c968c'), metalness: 1, roughness: 0.42, envMapIntensity: 0.55, bumpMap: windTex, bumpScale: 0.7 });   // brighter caught the glow pass: white neon
      } else {
        str.scale.set(1, 1, 1);
        Object.assign(m, { color: new THREE.Color('#8a847c'), metalness: 0.6, roughness: 0.5, envMapIntensity: 1, bumpMap: null });
      }
      m.needsUpdate = true;
    }
    const u = str.material.userData.u;
    if (u) {
      const hitAge = (ms - (v.laneHitMs[s] ?? -1e9)) / 1000;
      u.uAmp.value = real && hitAge < 0.8 ? 0.022 * Math.exp(-hitAge * 6) : 0; u.uAge.value = hitAge;
    }
    const on = held.has(s);
    str.position.y = 0.05 + (on ? Math.sin(ms / 16 + s) * 0.018 : 0);
    str.material.emissiveIntensity = (on ? 0.9 + 0.3 * Math.sin(ms / 50) : lanes.strGlow ?? 0.18) * (look.strings ? 0.12 : 1);   // real strings: just a hint of colour
  });
  const spark = (s, count, up) => {
    const cols = v.onFire ? ['#e0f2fe', '#38bdf8'] : [lanes.pads[s].col, '#ffe7a3'];
    for (let i = 0; i < count; i++) {
      const sp = newSpark(cols[i % 2]);
      sp.position.set(0, 0.3, laneZ(s));
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
      const sp = newSpark(cols[i]);
      sp.position.set(0, 0.3, laneZ(b.s));
      sparks.push({ sp, z0: laneZ(b.s), v: [0.6 + Math.random() * 1.2, 2.2 + Math.random() * 1.4, (Math.random() - 0.5) * 1.2], born: ms });
    }
  }
  seenBurst = Math.max(seenBurst, ...v.bursts.map((b) => b.at));
  for (let i = sparks.length - 1; i >= 0; i--) {
    const sk = sparks[i], a = (ms - sk.born) / 1000;
    if (a > 0.25) { sk.sp.visible = false; sparkPool.push(sk.sp); sparks.splice(i, 1); continue; }
    sk.sp.position.set(sk.v[0] * a, 0.3 + sk.v[1] * a - 6 * a * a, sk.z0 + sk.v[2] * a); sk.sp.material.opacity = 1 - a / 0.25;
  }

  composer.render();
}

// Sparks reuse their meshes (a new mesh and material for every spark, many a second while holding, kept the
// browser collecting garbage: short hitches, user: "the game still sort of skips sometimes for a tiny second").
const sparkPool = [];
function newSpark(color) {
  let sp = sparkPool.pop();
  if (!sp) { sp = new THREE.Mesh(mats.sparkGeo, new THREE.MeshBasicMaterial({ transparent: true, toneMapped: false })); world.add(sp); }
  sp.material.color.set(color); sp.visible = true;
  return sp;
}

// Before a song's first frame: every bar number and fret number texture it will need (also the grey ones for misses)
// made and sent to the GPU, a pool of notes built, and the shaders compiled with everything showing once. These
// used to be made the first time each was needed, mid-song (a miss made its grey number right then).
let warmFor = null;
function prewarm(v) {
  warmFor = v.notes;
  const ready = (key, make) => { if (!tex[key]) { tex[key] = make(); renderer.initTexture?.(tex[key]); } };
  v.bars.forEach((_, i) => ready(`bar${i}`, () => label(String(i + 1), '#d6c7b0', true)));
  for (const [, , s, f] of v.notes) {
    if (s >= lanes.n || typeof f !== 'number') continue;
    const open = f === 0;
    for (const missed of [false, true]) ready(`${f}|${open}|${s}|${missed}`, () => label(String(f), missed ? '#a1a1aa' : open ? v.colors[s % v.colors.length] : '#ffffff', !open && !missed));
  }
  while (pool.length + used.size < 60) pool.push(noteObject());
  for (let i = sparkPool.length; i < 40; i++) {
    const sp = new THREE.Mesh(mats.sparkGeo, new THREE.MeshBasicMaterial({ transparent: true, toneMapped: false }));
    sp.visible = false; world.add(sp); sparkPool.push(sp);
  }
  const spare = [...pool.flatMap((o) => [o.grp, o.tail, o.tailGlow, o.cap, o.refl]), ...sparkPool, ...lanes.pads.map((p) => p.pool)];
  for (const m of spare) m.visible = true;
  renderer.compile(scene, camera);
  for (const m of spare) m.visible = false;              // spares stay hidden until a note or spark takes them
}

window.Game3D = { mount, frame, padScreen, nameScreen, labels };
