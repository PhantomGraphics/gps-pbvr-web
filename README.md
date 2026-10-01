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
(calibration C0-C3, C3+R). `tests/cpp_reference` regenerates the C++ reference fixtures and needs the
Phantom C++ sources (this repo is meant to be checked out at `web/gps-pbvr` of the parent project).
