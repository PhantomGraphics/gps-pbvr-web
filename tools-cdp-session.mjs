// Drives the viewer page in a real browser via the DevTools protocol (serves web/ + fixtures).
// Phase 5 checks: session round trip, reproducibility (same seed -> identical PNG), device-lost recovery,
// trajectory ZIP. usage: node tools-cdp-session.mjs <browser.exe> <out-dir> [flags]
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { mkdtempSync, readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, extname, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('.', import.meta.url));
const [exe, out, ...extra] = process.argv.slice(2);
const types = { '.html': 'text/html', '.js': 'text/javascript', '.wasm': 'application/wasm' };
const srv = createServer((req, res) => {
  const u = decodeURIComponent(req.url.split('?')[0]);
  const f = u.startsWith('/fixtures/') ? join(root, 'tests', normalize(u)) : join(root, 'web', normalize(u === '/' ? '/index.html' : u));
  if (!existsSync(f)) { res.writeHead(404); return res.end(); }
  res.writeHead(200, { 'content-type': types[extname(f)] || 'application/octet-stream' });
  res.end(readFileSync(f));
}).listen(0);
const sport = srv.address().port;
const port = 9300 + Math.floor(Math.random() * 500);
const child = spawn(exe, [`--remote-debugging-port=${port}`, `--user-data-dir=${mkdtempSync(join(tmpdir(), 'gpsshot-'))}`, '--headless=new',
  '--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--no-first-run', '--window-size=1500,1000', ...extra, 'about:blank'], { stdio: 'ignore' });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let code = 1;
let exceptions = 0;
try {
  let targets;
  for (let i = 0; i < 50; i++) {
    try { targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json(); if (targets.length) break; } catch {}
    await sleep(200);
  }
  const ws = new WebSocket(targets.find((t) => t.type === 'page').webSocketDebuggerUrl);
  await new Promise((r) => (ws.onopen = r));
  let id = 0;
  const pending = new Map();
  ws.onmessage = (m) => {
    const d = JSON.parse(m.data);
    if (d.id && pending.has(d.id)) { pending.get(d.id)(d); pending.delete(d.id); }
    else if (d.method === 'Runtime.exceptionThrown') { exceptions++; console.log('PAGE EXCEPTION:', d.params.exceptionDetails.exception?.description || d.params.exceptionDetails.text); }
    else if (d.method === 'Runtime.consoleAPICalled' && ['error', 'warning'].includes(d.params.type)) console.log('PAGE', d.params.type + ':', d.params.args.map((a) => a.value ?? a.description).join(' '));
  };
  const send = (method, params = {}) => new Promise((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
  const ev = async (expr) => {
    const r = (await send('Runtime.evaluate', { expression: expr, returnByValue: true, awaitPromise: true })).result;
    return r?.exceptionDetails ? 'ERR ' + r.exceptionDetails.text : r?.result?.value;
  };
  await send('Runtime.enable');
  await send('Page.enable');
  for (let i = 0; i < 60 && !(await ev(`!!document.getElementById('stats')?.textContent.includes('累積')`)); i++) await sleep(500);
  // the light preset is the default; the check runs with it
  const results = [];
  const check = (name, ok, extra = '') => { results.push(ok); console.log(`${ok ? 'ok  ' : 'FAIL'} ${name}${extra ? ' | ' + extra : ''}`); };
  const open = async () => {
    await send('Page.navigate', { url: `http://127.0.0.1:${sport}/index.html` });
    for (let i = 0; i < 60 && !(await ev(`!!window.__gps`)); i++) await sleep(300);
  };
  const waitFor = async (expr, tries = 120) => { for (let i = 0; i < tries; i++) { if (await ev(expr)) return true; await sleep(250); } return false; };
  const loadFixture = () => ev(`(async () => {
    const b = new Uint8Array(await (await fetch('/fixtures/synthetic_sh1.ply')).arrayBuffer());
    await window.__gps.loadBytes(new File([b], 'synthetic_sh1.ply')); return 'loaded'; })()`);
  // FNV-1a over the PNG bytes of the canvas, taken after the accumulation target has been reached
  const hashNow = () => ev(`(async () => { const d = await window.__gps.canvasPng(); let h = 2166136261; for (const v of d) { h ^= v; h = Math.imul(h, 16777619) >>> 0; } return h.toString(16) + ':' + d.length; })()`);
  const settle = (n) => waitFor(`window.__gps.accumulated() >= ${n} && !window.__gps.isBusy()`);

  await open();
  console.log('fixture:', await loadFixture(), '|', await ev(`document.getElementById('msg').textContent`));
  const base = await ev(`(() => { const s = window.__gps.getSession(); s.settings.preset = 'custom'; s.settings.res = '320x180'; s.settings.target = 12;
    s.settings.seed = 7; s.settings.method = '2'; s.settings.calib = '3'; s.settings.radial = true; s.camera.yaw = 1.1; s.camera.pitch = 0.3; return JSON.stringify(s); })()`);
  await ev(`window.__gps.applySession(${base})`);
  check('session applies and converges', await settle(12));
  const h1 = await hashNow();
  const sess1 = await ev(`JSON.stringify(window.__gps.getSession())`);
  const s1 = JSON.parse(sess1);
  check('getSession round-trips settings/camera', s1.settings.seed === '7' && s1.settings.method === '2' && s1.settings.radial === true && Math.abs(s1.camera.yaw - 1.1) < 1e-9, `scene=${s1.scene.name} n=${s1.scene.gaussians}`);

  await ev(`window.__gps.loseDevice()`);
  check('device loss is detected and the viewer recovers', await waitFor(`document.getElementById('msg').textContent.includes('再初期化しました')`, 160));
  check('converges again after recovery', await settle(12));
  const h2 = await hashNow();
  check('image identical after device-loss recovery', h1 === h2, `${h1} vs ${h2}`);

  await open();
  await loadFixture();
  await ev(`window.__gps.applySession(${sess1})`);
  await settle(12);
  const h3 = await hashNow();
  check('image identical after page reload + session import (reproducible)', h1 === h3, `${h1} vs ${h3}`);

  await ev(`(() => { const s = window.__gps.getSession(); s.settings.seed = 8; window.__gps.applySession(s); })()`);
  await settle(12);
  const h4 = await hashNow();
  check('different seed gives a different image', h4 !== h1, `${h4}`);

  check('invalid session is rejected', (await ev(`(() => { try { window.__gps.applySession({ format: 'nope' }); return 'accepted'; } catch (e) { return 'rejected'; } })()`)) === 'rejected');

  await ev(`window.__gps.applySession(${sess1})`);
  await settle(12);
  const zipB64 = await ev(`(async () => { const blob = await window.__gps.renderTrajectory(window.__gps.orbitKeys(), 3);
    const u = new Uint8Array(await blob.arrayBuffer()); let s = ''; for (let i = 0; i < u.length; i += 8192) s += String.fromCharCode(...u.subarray(i, i + 8192)); return btoa(s); })()`);
  mkdirSync(out, { recursive: true });
  const zipPath = join(out, 'trajectory.zip');
  writeFileSync(zipPath, Buffer.from(zipB64 || '', 'base64'));
  const { spawnSync } = await import('node:child_process');
  const py = spawnSync('python', [join(root, 'tools-check-zip.py'), zipPath], { encoding: 'utf-8' });
  check('trajectory ZIP is valid (3 PNGs: orbit start == end, middle differs; stats.csv; session.json)', py.status === 0, (py.stdout + py.stderr).trim().split('\n').slice(-2).join(' '));
  check('no page exceptions', exceptions === 0, `${exceptions}`);
  code = results.every(Boolean) ? 0 : 1;
  console.log(code === 0 ? 'PHASE5: PASS' : 'PHASE5: FAIL');
  ws.close();
} finally { child.kill(); srv.close(); }
process.exit(code);
