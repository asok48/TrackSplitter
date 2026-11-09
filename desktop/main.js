// Encore as its own Mac app (user: "how much of a lift would it be to make this into its own app?" -> "yes let's do 1").
// Electron runs Chrome's engine, so the bass input (both of the Scarlett's channels), Web Audio and the timing
// calibration behave exactly as in Chrome (a WebKit wrapper was turned down for that reason when Encore became a Chrome
// app). It shows /game in its own window, starts Track Splitter's server when it isn't running (and stops it on quit if
// it started it), lets the page use the microphone, and lets REC record this window only: no screen picker and no
// screen-recording permission, so nothing outside Encore can end up in a video. Loaded by the stub inside
// ~/Applications/Encore.app (desktop/make-app.sh), so updates to this file apply on the next start.
const { app, BrowserWindow, session, systemPreferences, shell, ipcMain } = require('electron');
const { spawn } = require('child_process');
const fs = require('fs');
const os = require('os');
const path = require('path');
const http = require('http');

const ROOT = path.resolve(__dirname, '..');
const ORIGIN = 'http://127.0.0.1:7860';
const LOG = path.join(os.homedir(), 'Library', 'Logs', 'Track Splitter.log');
const ALLOWED = ['media', 'display-capture', 'fullscreen', 'clipboard-sanitized-write'];
const ours = (url) => { try { return new URL(url).origin === ORIGIN; } catch { return false; } };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

app.setName('Encore');
let win = null, server = null, quitting = false;

const isUp = () => new Promise((resolve) => {
  const req = http.get(`${ORIGIN}/api/info`, (res) => { res.resume(); resolve(res.statusCode === 200); });
  req.on('error', () => resolve(false));
  req.setTimeout(800, () => { req.destroy(); resolve(false); });
});

// start.sh sets up / updates the packages first (the first run takes minutes), then runs app.py without a browser.
async function startServer() {
  if (await isUp()) return true;
  const out = fs.openSync(LOG, 'a');
  server = spawn('/bin/sh', ['start.sh', '--no-browser'], {
    cwd: ROOT, stdio: ['ignore', out, out],
    env: { ...process.env, PATH: `/opt/homebrew/bin:/usr/local/bin:${process.env.PATH || '/usr/bin:/bin:/usr/sbin:/sbin'}` },   // (opened from Finder the PATH is minimal)
  });
  server.on('exit', () => { server = null; });
  for (let i = 0; i < 1500; i++) {                     // up to 10 minutes on a first run
    if (await isUp()) return true;
    if (!server) return false;
    await sleep(400);
  }
  return false;
}

const message = (text) => `data:text/html;charset=utf-8,${encodeURIComponent(
  `<html style="height:100%;margin:0;background:#05060a;color:#a8a29e;font:600 15px -apple-system,system-ui;display:grid;place-items:center"><body>${text}</body></html>`)}`;

if (!app.requestSingleInstanceLock()) app.quit();
else {
  app.on('second-instance', () => { if (win) { if (win.isMinimized()) win.restore(); win.focus(); } });
  app.whenReady().then(async () => {
    const s = session.defaultSession;
    s.setPermissionRequestHandler((wc, permission, done, details) => done(ours(details.requestingUrl || wc.getURL()) && ALLOWED.includes(permission)));
    s.setPermissionCheckHandler((wc, permission, origin) => ours(origin) && ALLOWED.includes(permission));
    // REC: the page's getDisplayMedia records its own window, without asking
    s.setDisplayMediaRequestHandler((request, done) => (ours(request.securityOrigin) && request.frame ? done({ video: request.frame }) : done({})));
    if (process.platform === 'darwin') {
      systemPreferences.askForMediaAccess('microphone').catch(() => {});   // the Mac asks once, for Encore
      app.dock?.setIcon(path.join(ROOT, 'static', 'icons', 'encore-512.png'));
    }
    win = new BrowserWindow({
      width: 1440, height: 900, minWidth: 1024, minHeight: 700, backgroundColor: '#05060a', title: 'Encore',
      webPreferences: { preload: path.join(__dirname, 'preload.js'), backgroundThrottling: false, contextIsolation: true, sandbox: true, autoplayPolicy: 'no-user-gesture-required' },   // sound without a click first (previews, the audio clock)
    });
    win.on('page-title-updated', (e) => e.preventDefault());
    win.webContents.setWindowOpenHandler(({ url }) => { if (ours(url)) return { action: 'allow' }; shell.openExternal(url); return { action: 'deny' }; });
    win.webContents.on('will-navigate', (e, url) => { if (!ours(url) && !url.startsWith('data:')) { e.preventDefault(); shell.openExternal(url); } });
    win.loadURL(message('Starting Track Splitter…'));
    if (await startServer()) win.loadURL(`${ORIGIN}/game`);
    else win.loadURL(message('Track Splitter didn’t start. The log is in ~/Library/Logs/Track Splitter.log.'));
  });
}

ipcMain.on('encore-quit', () => app.quit());           // Encore's own Quit (the page has stopped the server already)
app.on('window-all-closed', () => app.quit());
// Quitting stops the server if this app started it (one started some other way keeps running).
app.on('before-quit', (e) => {
  if (!server || quitting) return;
  quitting = true;
  e.preventDefault();
  const req = http.request(`${ORIGIN}/api/quit`, { method: 'POST' });
  req.on('error', () => {});
  req.end();
  setTimeout(() => { server?.kill(); app.quit(); }, 1500);
});
