// Trusted input events through the DevTools protocol (not synthetic JS events): left / right / middle / Shift drag,
// the pan-mode toggle, arrow keys and a two-finger touch pan. Exit 0 iff every expectation holds.
// usage: [SITE=web|dist] node tools-cdp-input.mjs <browser.exe>
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { mkdtempSync, readFileSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, extname, normalize } from 'node:path';
import { fileURLToPath } from 'node:url';
const root = fileURLToPath(new URL('.', import.meta.url));
let failed = 0;
const expect = (name, ok) => { if (!ok) { failed++; console.log('FAIL', name); } };
const SITE = process.env.SITE || 'web';
const srv = createServer((req, res) => {
  const u = decodeURIComponent(req.url.split('?')[0]);
  const f = join(root, SITE, normalize(u === '/' ? '/index.html' : u));
  if (!existsSync(f)) { res.writeHead(404); return res.end(); }
  res.writeHead(200, { 'content-type': { '.html': 'text/html', '.js': 'text/javascript', '.wasm': 'application/wasm' }[extname(f)] || 'application/octet-stream' });
  res.end(readFileSync(f));
}).listen(0);
const sport = srv.address().port, port = 9300 + Math.floor(Math.random() * 500);
const child = spawn(process.argv[2], [`--remote-debugging-port=${port}`, `--user-data-dir=${mkdtempSync(join(tmpdir(), 'rm-'))}`, '--headless=new', '--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--no-first-run', '--window-size=1500,1000', 'about:blank'], { stdio: 'ignore' });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let targets; for (let i = 0; i < 50; i++) { try { targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json(); if (targets.length) break; } catch {} await sleep(200); }
const ws = new WebSocket(targets.find((t) => t.type === 'page').webSocketDebuggerUrl); await new Promise((r) => (ws.onopen = r));
let id = 0; const pending = new Map();
ws.onmessage = (m) => { const d = JSON.parse(m.data); if (d.id && pending.has(d.id)) { pending.get(d.id)(d); pending.delete(d.id); } };
const send = (method, params = {}) => new Promise((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
const ev = async (e) => (await send('Runtime.evaluate', { expression: e, returnByValue: true, awaitPromise: true })).result?.result?.value;
ws.addEventListener('message', (m) => { const d = JSON.parse(m.data); if (d.method === 'Runtime.exceptionThrown') console.log('PAGE EXC', d.params.exceptionDetails.exception?.description || d.params.exceptionDetails.text); if (d.method === 'Runtime.consoleAPICalled' && d.params.type === 'error') console.log('PAGE ERR', d.params.args.map((a) => a.value ?? a.description).join(' ')); });
await send('Runtime.enable'); await send('Page.enable'); await send('Page.navigate', { url: `http://127.0.0.1:${sport}/index.html` });
for (let i = 0; i < 60 && !(await ev('!!window.__gps')); i++) await sleep(300);
const rect = JSON.parse(await ev(`JSON.stringify(document.getElementById('cv').getBoundingClientRect())`));
const cx = rect.x + rect.width / 2, cy = rect.y + rect.height / 2;
const cam = async () => JSON.parse(await ev(`JSON.stringify(window.__gps.getSession().camera)`));
const mouse = (type, x, y, button, buttons, modifiers = 0) => send('Input.dispatchMouseEvent', { type, x, y, button, buttons, modifiers, clickCount: type === 'mouseMoved' ? 0 : 1 });
const run = async (name, button, buttons, modifiers) => {
  const c0 = await cam();
  await mouse('mousePressed', cx, cy, button, buttons, modifiers);
  for (let i = 1; i <= 10; i++) { await mouse('mouseMoved', cx + i * 8, cy + i * 4, button, buttons, modifiers); await sleep(30); }
  await mouse('mouseReleased', cx + 80, cy + 40, button, 0, modifiers);
  await sleep(200);
  const c1 = await cam();
  const dT = Math.hypot(...c1.target.map((v, k) => v - c0.target[k])), dY = Math.abs(c1.yaw - c0.yaw);
  console.log(name.padEnd(22), 'dTarget', dT.toFixed(4), 'dYaw', dY.toFixed(4));
  expect(name, name === 'left drag' ? dY > 0.05 && dT < 1e-9 : dT > 1e-3 && dY < 1e-9);
};
await run('left drag', 'left', 1, 0);
await run('right drag', 'right', 2, 0);
await run('middle drag', 'middle', 4, 0);
await run('shift + left drag', 'left', 1, 8);
const panmode = async () => { await ev(`document.getElementById('panmode').click()`); };
await panmode(); await run('pan mode + left drag', 'left', 1, 0); await panmode();
// arrow key
{ const c0 = await cam(); await send('Input.dispatchKeyEvent', { type: 'keyDown', key: 'ArrowRight', code: 'ArrowRight', windowsVirtualKeyCode: 39 }); await send('Input.dispatchKeyEvent', { type: 'keyUp', key: 'ArrowRight', code: 'ArrowRight', windowsVirtualKeyCode: 39 }); await sleep(100);
  const c1 = await cam(); const dT = Math.hypot(...c1.target.map((v, k) => v - c0.target[k])); console.log('arrow key'.padEnd(22), 'dTarget', dT.toFixed(4)); expect('arrow key', dT > 1e-3 && Math.abs(c1.yaw - c0.yaw) < 1e-9); }
// two-finger touch: pan + pinch
{ const c0 = await cam(); const T = (type, pts) => send('Input.dispatchTouchEvent', { type, touchPoints: pts.map((p, i) => ({ x: p[0], y: p[1], id: i + 1 })) });
  await T('touchStart', [[cx - 40, cy], [cx + 40, cy]]);
  for (let i = 1; i <= 8; i++) { await T('touchMove', [[cx - 40 + i * 5, cy + i * 3], [cx + 40 + i * 5, cy + i * 3]]); await sleep(30); }
  await T('touchEnd', []); await sleep(200);
  const c1 = await cam(); const dT = Math.hypot(...c1.target.map((v, k) => v - c0.target[k])); console.log('two-finger drag'.padEnd(22), 'dTarget', dT.toFixed(4)); expect('two-finger drag', dT > 1e-3 && Math.abs(c1.yaw - c0.yaw) < 1e-9); }
console.log('env:', (await ev(`document.getElementById('env').textContent`)).slice(-40));
console.log(failed ? 'INPUT: FAIL' : 'INPUT: PASS');
child.kill(); srv.close(); process.exit(failed ? 1 : 0);
