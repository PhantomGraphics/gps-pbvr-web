// Launches Edge/Chrome with WebGPU enabled, loads the Phase 0 page, waits for the
// result via the DevTools protocol and prints the report. Exit 0 iff PASS.
// usage: node tools-cdp-check.mjs <browser.exe> <url> [extra flags...]
import { spawn } from 'node:child_process';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
const [exe, url, ...extra] = process.argv.slice(2);
const port = 9300 + Math.floor(Math.random() * 500);
const prof = mkdtempSync(join(tmpdir(), 'gpscdp-'));
const child = spawn(exe, [`--remote-debugging-port=${port}`, `--user-data-dir=${prof}`, '--headless=new',
  '--enable-unsafe-webgpu', '--ignore-gpu-blocklist', '--no-first-run', ...extra, 'about:blank'], { stdio: 'ignore' });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let code = 3;
try {
  let targets;
  for (let i = 0; i < 50; i++) {
    try { targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json(); if (targets.length) break; } catch {}
    await sleep(200);
  }
  const page = targets.find((t) => t.type === 'page');
  const ws = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((r) => (ws.onopen = r));
  let id = 0; const pending = new Map();
  ws.onmessage = (m) => {
    const d = JSON.parse(m.data);
    if (d.id && pending.has(d.id)) { pending.get(d.id)(d); pending.delete(d.id); }
    else if (d.method === 'Runtime.exceptionThrown') console.log('PAGE EXCEPTION:', d.params.exceptionDetails.exception?.description || d.params.exceptionDetails.text);
    else if (d.method === 'Runtime.consoleAPICalled' && ['error', 'warning'].includes(d.params.type)) console.log('PAGE CONSOLE', d.params.type + ':', d.params.args.map((a) => a.value ?? a.description).join(' '));
  };
  const send = (method, params = {}) => new Promise((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
  const ev = async (expr) => (await send('Runtime.evaluate', { expression: expr, returnByValue: true })).result?.result?.value;
  await send('Runtime.enable'); await send('Page.enable'); await send('Page.navigate', { url });
  const t0 = Date.now();
  let res;
  while (Date.now() - t0 < (Number(process.env.CDP_TIMEOUT_MS) || 180000)) {
    await sleep(1000);
    res = await ev(`document.body && document.body.dataset.result || (document.getElementById('status')?.textContent.startsWith('FAIL')?'fail':'')`);
    if (res) break;
  }
  console.log('UA:', await ev('navigator.userAgent'));
  console.log(await ev(`document.getElementById('report')?.textContent || ''`));
  console.log('status:', await ev(`document.getElementById('status')?.textContent`));
  code = res === 'pass' ? 0 : 1;
  ws.close();
} finally { child.kill(); }
process.exit(code);
