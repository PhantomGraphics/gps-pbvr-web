// Drives the viewer page in a real browser via the DevTools protocol: serves web/ (+ fixtures),
// loads a PLY through the file input, orbits with pointer events, waits for accumulation and saves
// a screenshot. Exit 0 unless the page reports an internal inconsistency or an exception.
// usage: node tools-cdp-shot.mjs <browser.exe> <out.png> [extra browser flags...]
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
  console.log('default preset:', await ev(`document.getElementById('preset').value + ' ' + document.getElementById('res').value + ' spp' + document.getElementById('spp').value`));
  // slow the convergence down so a history reset is observable (the light preset converges in ~0.5 s)
  await ev(`(() => { const t = document.getElementById('target'); t.value = '600'; t.dispatchEvent(new Event('change')); })()`);
  const stat = () => ev(`document.getElementById('stats').textContent`);
  const yawOf = (t) => { const m = /yaw (-?[\d.]+)/.exec(t); return m ? +m[1] : NaN; };
  const accOf = (t) => { const m = /累積 (\d+)\//.exec(t); return m ? +m[1] : -1; };
  const waitAcc = async (n) => {
    for (let i = 0; i < 160; i++) { const t = await stat(); if (accOf(t) >= n) return t; await sleep(500); }
    return await stat();
  };
  console.log('--- demo scene\n' + (await waitAcc(16)));
  console.log(await ev(`(async () => {
    const b = new Uint8Array(await (await fetch('/fixtures/synthetic_sh1.ply')).arrayBuffer());
    const dt = new DataTransfer(); dt.items.add(new File([b], 'synthetic_sh1.ply'));
    const i = document.getElementById('file'); i.files = dt.files; i.dispatchEvent(new Event('change'));
    await new Promise((r) => setTimeout(r, 400));
    return 'msg: ' + document.getElementById('msg').textContent;
  })()`));
  console.log('--- after PLY load\n' + (await waitAcc(48)));

  const yaw0 = yawOf(await stat());
  if (process.env.PROBE_RESET) {
    await waitAcc(40);
    const t0 = await stat();
    await ev(`document.getElementById('reset').click()`);
    const samples = [];
    for (let i = 0; i < 6; i++) { await sleep(400); samples.push(accOf(await stat())); }
    console.log(`acc before reset=${accOf(t0)}; after click, every 400ms: ${samples.join(', ')}; yaw ${yawOf(t0)} -> ${yawOf(await stat())}`);
    process.exit(0);
  }
  if (process.env.PROBE) {
    const { createHash } = await import('node:crypto');
    for (let i = 0; i < 6; i++) {
      await sleep(2500);
      const t = await stat();
      const a = (await send('Page.captureScreenshot', { format: 'png', clip: { x: 16, y: 64, width: 1100, height: 620, scale: 0.5 } })).result.data;
      await sleep(600);
      const b = (await send('Page.captureScreenshot', { format: 'png', clip: { x: 16, y: 64, width: 1100, height: 620, scale: 0.5 } })).result.data;
      console.log(`step ${i}: yaw=${yawOf(t)} acc=${accOf(t)} shot=${createHash('md5').update(a).digest('hex').slice(0, 8)} shot2=${createHash('md5').update(b).digest('hex').slice(0, 8)}`);
      await ev(`(() => { const c = document.getElementById('cv'); const r = c.getBoundingClientRect(); const x = r.left + r.width / 2, y = r.top + r.height / 2;
        const pe = (t, dx) => c.dispatchEvent(new PointerEvent(t, { clientX: x + dx, clientY: y, pointerId: 1, bubbles: true }));
        pe('pointerdown', 0); pe('pointermove', 120); pe('pointerup', 120); })()`);
    }
    process.exit(0);
  }
  if (process.env.METHODS) {
    // Phase 3: switch the particle model in the UI and capture each converged image.
    const set = (id, v) => ev("(() => { const e = document.getElementById('" + id + "'); if (e.type === 'checkbox') e.checked = " + JSON.stringify(v) + "; else e.value = " + JSON.stringify(String(v)) + "; e.dispatchEvent(new Event('change')); })()");
    let bad = exceptions > 0;
    for (const [name, m, calib, radial] of [['gps', 0, 3, false], ['prop', 1, 3, false], ['ext_c3', 2, 3, false], ['ext_c3r', 2, 3, true], ['vc', 3, 3, false]]) {
      await set('method', m); await set('calib', calib); await set('radial', radial);
      if (m === 3) await set('basek', 4096);
      await sleep(600);
      const t = await waitAcc(32);
      const file = out.replace(/\.png$/, '_' + name + '.png');
      writeFileSync(file, Buffer.from((await send('Page.captureScreenshot', { format: 'png' })).result.data, 'base64'));
      const keep = t.split('\n').filter((l) => /累積|粒子数|方式|⚠/.test(l)).join('\n');
      console.log('--- ' + name + '\n' + keep + '\n -> ' + file);
      if (/内部不整合|届かない|打切り/.test(t)) bad = true;
    }
    code = bad || exceptions > 0 ? 1 : 0;
    ws.close();
    throw Object.assign(new Error('done'), { methodsDone: true });
  }
  const shotBefore = (await send('Page.captureScreenshot', { format: 'png' })).result.data;
  const before = accOf(await stat());
  await ev(`(() => {
    const c = document.getElementById('cv'); const r = c.getBoundingClientRect();
    const x = r.left + r.width / 2, y = r.top + r.height / 2;
    const pe = (t, dx) => c.dispatchEvent(new PointerEvent(t, { clientX: x + dx, clientY: y, pointerId: 1, bubbles: true }));
    pe('pointerdown', 0); pe('pointermove', 80);
  })()`);
  await sleep(1200); // the stats panel refreshes every 500 ms
  const after = accOf(await stat());
  console.log(`accumulation before orbit=${before}, shortly after=${after} -> history reset: ${after >= 0 && after < before}`);
  const yaw1 = yawOf(await stat());
  console.log(`yaw before=${yaw0} after=${yaw1} (expected change about -0.40)`);
  const final = await waitAcc(64);
  await ev(`document.getElementById('reset').click()`);
  await sleep(1200);
  const yaw2 = yawOf(await stat());
  console.log(`after reset button: yaw=${yaw2} (home ${yaw0}) acc=${accOf(await stat())}`);
  const resetOk = Math.abs(yaw2 - yaw0) < 1e-6;
  const shotAfter = (await send('Page.captureScreenshot', { format: 'png' })).result.data;
  const moved = shotBefore !== shotAfter;
  console.log(`canvas changed after orbit: ${moved}`);
  console.log('--- converged after orbit\n' + final);
  const shot = await send('Page.captureScreenshot', { format: 'png' });
  writeFileSync(out, Buffer.from(shot.result.data, 'base64'));
  console.log('screenshot ->', out);
  code = /内部不整合/.test(final) || exceptions > 0 || !(after >= 0 && after < before) || !(Math.abs(yaw1 - yaw0) > 0.2) || !resetOk ? 1 : 0;
  ws.close();
} catch (e) { if (!e.methodsDone) { console.log('ERROR', e); code = 1; } } finally { child.kill(); srv.close(); }
process.exit(code);
