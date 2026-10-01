# gps-pbvr-web

Gaussian Point Splatting (GPS) and PBVR-style particle rendering of 3D Gaussian Splatting scenes in the browser,
written in Rust (WebAssembly) with wgpu / WebGPU compute shaders (WGSL).

- `crates/gps-core` – math, camera, CPU reference oracle
- `crates/gps-io` – 3DGS PLY loader (ASCII / binary, SH 0-3)
- `crates/gps-render` – wgpu renderer, WGSL shaders in `shaders/`, GPU-vs-oracle verification
- `crates/gps-web` – wasm-bindgen viewer
- `crates/gps-verify` – native verification / benchmark CLI
- `web/` – static viewer (`index.html`) and self-check page (`check.html`)

## Build and run

Requires Rust (see `rust-toolchain.toml`, with the `wasm32-unknown-unknown` target) and `wasm-bindgen-cli`.

```powershell
./build-web.ps1                  # builds web/pkg
cd web; python -m http.server 8765 --bind 127.0.0.1
# open http://127.0.0.1:8765/index.html in Chrome / Edge (WebGPU)
```

Native checks: `cargo test --release --workspace` and `cargo run --release -p gps-verify -- all`.
Browser checks: open `web/check.html` (or `node tools-cdp-check.mjs <browser.exe> <url>`).

WebGPU needs HTTPS or localhost. Methods: GPS, PBVR Proportional / Extinction / ViewConditioned
(calibration C0-C3, C3+R).

`tests/fixtures/cpp_*.f64` are outputs of the original C++ implementation, checked in so that
`cargo test` (the `cpp_crosscheck` test) is self-contained. They are generated and owned by the
Crystal2024 parent project, not by this repository.

## Reproducibility and output

- **Session JSON** ("設定を保存/読み込み"): all settings, the orbit camera and the seed. Same data + same session = the same
  image (checked by `tools-cdp-session.mjs`: identical PNG after a page reload and after device-loss recovery).
- **Outputs**: PNG, stats (JSON / CSV), and a camera trajectory exported as one ZIP (PNG sequence + `stats.csv` +
  `session.json`), either from keyframes or a 360-degree orbit.
- **Robustness**: a lost GPU device is detected, the viewer is recreated and the scene and session restored.

## Static distribution

`./build-dist.ps1` writes `dist/` (`index.html`, `check.html`, `pkg/gps_web.js`, `pkg/gps_web_bg.wasm`). Host it over
HTTPS with `.wasm` served as `application/wasm`. Without WebGPU the page shows an explanation; tested on Chrome / Edge 154
(Windows, integrated GPU). Firefox / Safari are not verified yet.

### GitHub Pages

`.github/workflows/pages.yml` builds and deploys `dist/` on every push to `main` (it also runs `cargo test` first).
One-time setup: repository **Settings -> Pages -> Source: GitHub Actions**. The site is then served at
`https://<owner>.github.io/<repo>/`. The workflow pins `wasm-bindgen-cli` to the version in `Cargo.lock`
(`WASM_BINDGEN_VERSION`); update both together.

## Browser tests

`node tools-cdp-check.mjs <browser.exe> <url>/check.html` (GPU-vs-CPU checks), `node tools-cdp-session.mjs <browser.exe> <out-dir>`
(session / reproducibility / device loss / trajectory ZIP; needs `python` for `tools-check-zip.py`),
`node tools-cdp-input.mjs <browser.exe>` (trusted mouse / key / touch input: orbit, pan, pinch).

Layout: the side panel can be closed (button or `P`) and the viewer then fills the window; `F` / the button toggles full screen
(a status line stays on the image). The choice is remembered.

Navigation: left drag = orbit; right / middle drag, Shift+drag, arrow keys or the "移動モード" toggle = pan; wheel = zoom;
two fingers = pan + pinch zoom. The page shows its build stamp next to the user agent (the js/wasm pair is loaded with a
matching `?v=` so a stale cache cannot mix versions).

## Benchmark on a real file

`cargo run --release -p gps-verify -- bench-ply <file.ply> [WxH] [spp_side]` prints GPU time per ensemble for GPS and PBVR at
three camera distances (`MAXPTS=<n>` sets the per-splat particle cap).
