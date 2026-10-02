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

Experimental half-resolution mode: select **描画スケール（実験） → 半解像度＋双線形拡大**.
The canvas/export resolution stays unchanged; particle generation and accumulation use half
the width and height. Density and opacity are not reduced. The composite filters linear RGB
before exposure and gamma. This baseline uses bilinear interpolation, not depth-guided
reconstruction, so thin structures and silhouettes can soften. Scale changes reset history;
the selected scale is saved in the session. Full resolution remains the default.

Compare a real PLY at the framed and close-up cameras with identical PBVR Extinction C3+R
settings (64 ensembles, SH unchanged): set `REAL=<path>`, `RES=1280x720`, and `SCALE_COMPARE=1`,
then run `node tools-cdp-real.mjs <browser.exe> <out-prefix>`. The tool writes four full-size
canvas PNGs and a comparison JSON with camera/settings, counters and convergence wall time
(browser presentation/polling included; not an isolated GPU timestamp).

Trial on `train_point_cloud.ply` (559,263 Gaussians), Intel integrated graphics:
native PBVR Extinction C3+R, SPP=1, cap=16,384, six synchronised ensembles after
warm-up, 1280x720 versus 640x360 internal resolution:

| Camera | Full (ms/ensemble) | Half (ms/ensemble) |
|---|---:|---:|
| Near | 203.6 | 117.5 |
| Mid | 70.8 | 36.9 |
| Far | 20.7 | 12.2 |

These are CPU submission + GPU completion times, excluding display composite,
not GPU timestamps. Both modes hit the existing per-splat cap in the near/mid
views; this is a capped practical comparison, not a proof of equal opacity or
unbiased convergence. Browser captures with cap=65,536 and 64 ensembles preserve
the full 1280x720 canvas and show softer lettering/railings in half mode. The
mode is manually selected; automatic full-resolution refinement after motion
and depth-guided upsampling are not part of this initial experiment.

`cargo run --release -p gps-verify -- bench-ply <file.ply> [WxH] [spp_side]` prints GPU time per ensemble for GPS and PBVR at
three camera distances (`MAXPTS=<n>` sets the per-splat particle cap).

### Clipped screen occupancy (experimental particle reduction)

Select `画面空間・被覆判定（粒子削減）` / `Method::ScreenOccupancy` (index 4).
The method clips each projected footprint to the viewport before creating work.
Faint footprints use uniform Poisson candidates with radial extinction rejection;
dense footprints use one Bernoulli coverage test per subpixel. Both implement
`P(hit) = 1 - exp(density * log(1 - alpha))`, where
`alpha = opacity * exp(-r²/2)`. Density is not reduced to meet a budget.
The per-splat particle cap is deliberately ignored; scan overflow remains detected.
The footprint drops tails below 1/255 coverage. It uses projected covariance
(including low-pass filtering) and splat-centre depth, rather than world-space
3D particle depth. Calibration, radial/centre/jitter switches do not apply.
It is a projected approximation, not an interchangeable 3D PBVR estimator.

Train, 559,263 Gaussians, SH3, Intel integrated GPU, 1280x720, SPP1:

| View | Extinction C3+R ms/ensemble | Screen occupancy ms/ensemble |
|---|---:|---:|
| Near | 207.5 | 152.0 |
| Mid | 73.3 | 86.7 |
| Far | 20.7 | 39.4 |

CPU submission + GPU completion, six ensembles after warmup, excluding composite.
Near generated-point counters fall from 57,886,950 to 5,884,944; the old counter
includes out-of-viewport particles, while occupancy counts accepted visible hits.
Occupancy still executes 80,558,539 candidate tests per ensemble. Existing PBVR
hits the 16,384 per-splat cap; occupancy reports no truncation, skipped ensembles,
or orphan subpixels. The method helps this near view but is slower at mid/far
distances, so it remains opt-in. It can be combined with half-resolution rendering.
GPU checks compare occupancy against analytic projected alpha and verify sparse
and dense stream determinism. `bench-ply` now includes this method.
