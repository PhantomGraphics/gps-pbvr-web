// Drives the viewer page in a real browser via the DevTools protocol: serves web/ (+ fixtures),
// loads a PLY through the file input, orbits with pointer events, waits for accumulation and saves
// a screenshot. Exit 0 unless the page reports an internal inconsistency or an exception.
// usage: REAL=<file.ply> node tools-cdp-real.mjs <browser.exe> <out-prefix> [flags]
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { mkdtempSync, readFileSync, writeFileSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, extname, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = fileURLToPath(new URL('.', import.meta.url));
const [exe, out, ...extra] = process.argv.slice(2);
const types = { '.html': 'text/html', '.js': 'text/javascript', '.wasm': 'application/wasm' };
const srv = createServer((req, res) => {
  const u = decodeURIComponent(req.url.split('?')[0]);
  if (u === '/__real') { res.writeHead(200, { 'content-type': 'application/octet-stream' }); return res.end(readFileSync(process.env.REAL)); }
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
  await send('Page.navigate', { url: `http://127.0.0.1:${sport}/index.html` });
  for (let i = 0; i < 60 && !(await ev(`!!document.getElementById('stats')?.textContent.includes('累積')`)); i++) await sleep(500);
  // the light preset is the default; the check runs with it
  const stat = () => ev(`document.getElementById('stats').textContent`);
  const accOf = (t) => { const m = /累積 (\d+)\//.exec(t); return m ? +m[1] : -1; };
  const waitAcc = async (n) => { for (let i = 0; i < 1200; i++) { const t = await stat(); if (accOf(t) >= n) return t; await sleep(500); } return await stat(); };
  const set = (id, v) => ev("(() => { const e = document.getElementById('" + id + "'); if (e.type === 'checkbox') e.checked = " + JSON.stringify(v) + "; else e.value = " + JSON.stringify(String(v)) + "; e.dispatchEvent(new Event('change')); })()");
  await set('target', 64);
  if (process.env.LOD) await set('lod', 'adaptive');
  console.log(await ev(`(async () => {
    const t0 = performance.now();
    const b = new Uint8Array(await (await fetch('/__real')).arrayBuffer());
    const t1 = performance.now();
    await window.__gps.loadBytes(new File([b], 'real.ply'));
    return 'fetch ' + Math.round(t1 - t0) + ' ms, parse+upload ' + Math.round(performance.now() - t1) + ' ms | msg: ' + document.getElementById('msg').textContent + ' | sh=' + document.getElementById('sh').value;
  })()`));
  const yd = process.env.YDOWN;
  if (yd !== undefined) await set('ydown', yd === '1');
  if (process.env.MOTION) {
    // Orbit with pointer events for ~3 s and record the frame intervals (moving), then wait for convergence (still).
    const measure = (ms) => ev(`(async () => { const c = document.getElementById('cv'); const r = c.getBoundingClientRect(); const x = r.left + r.width / 2, y = r.top + r.height / 2;
      const pe = (t, dx) => c.dispatchEvent(new PointerEvent(t, { clientX: x + dx, clientY: y, pointerId: 1, bubbles: true }));
      const iv = []; let last = performance.now(); const t0 = last; pe('pointerdown', 0); let dx = 0;
      while (performance.now() - t0 < ${ms}) { await new Promise((r) => requestAnimationFrame(r)); const n = performance.now(); iv.push(n - last); last = n; dx += 6; pe('pointermove', dx); }
      pe('pointerup', dx); iv.sort((a, b) => a - b);
      return JSON.stringify({ frames: iv.length, p50: iv[Math.floor(iv.length * 0.5)], p95: iv[Math.floor(iv.length * 0.95)], moving: document.getElementById('stats').textContent.includes('移動中') }); })()`);
    for (const lod of ['manual', 'adaptive']) {
      await set('lod', lod); await sleep(1500); await waitAcc(8);
      console.log('orbit (' + lod + '): ' + await measure(3000));
      const t0 = Date.now(); const t = await waitAcc(64);
      console.log('  converged to 64 in ' + (Date.now() - t0) + ' ms after the motion stopped; ' + t.split('\n').filter((l) => /累積|⚠|移動中/.test(l)).join(' | '));
    }
  }
  let bad = exceptions > 0;
  for (const [name, m, calib, radial, shv] of [['gps_sh3', 0, 3, false, 3], ['gps_sh0', 0, 3, false, 0], ['ext_c3r_sh3', 2, 3, true, 3]]) {
    await set('method', m); await set('calib', calib); await set('radial', radial); await set('sh', shv);
    await sleep(800);
    const t = await waitAcc(64);
    const file = out + '_' + name + '.png';
    writeFileSync(file, Buffer.from((await send('Page.captureScreenshot', { format: 'png' })).result.data, 'base64'));
    console.log('--- ' + name + '\n' + t + '\n -> ' + file);
    if (/内部不整合|打切り/.test(t)) bad = true;
  }
  code = bad || exceptions > 0 ? 1 : 0;
  ws.close();
} finally { child.kill(); srv.close(); }
process.exit(code);
