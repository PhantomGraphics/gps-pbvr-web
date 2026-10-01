# Builds gps-web to web/pkg (static site; serve the web/ directory over http(s)).
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    cargo build -p gps-web --target wasm32-unknown-unknown --release
    if ($LASTEXITCODE) { throw 'cargo build failed' }
    wasm-bindgen --target web --out-dir web/pkg target/wasm32-unknown-unknown/release/gps_web.wasm
    if ($LASTEXITCODE) { throw 'wasm-bindgen failed' }
} finally { Pop-Location }
