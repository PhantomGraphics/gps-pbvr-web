# Builds a static distribution in dist/ (copy it to any HTTPS static host; CI publishes it to GitHub Pages).
# Contains only what the page loads: index.html, check.html, pkg/gps_web.js, pkg/gps_web_bg.wasm.
# Works with Windows PowerShell and pwsh on Linux/macOS.
param([switch]$SkipBuild)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    if (-not $SkipBuild) { & (Join-Path $PSScriptRoot 'build-web.ps1') }
    $dist = Join-Path $PSScriptRoot 'dist'
    $pkg = Join-Path $dist 'pkg'
    if (Test-Path $dist) { Remove-Item -Recurse -Force $dist }
    New-Item -ItemType Directory -Force $pkg | Out-Null
    foreach ($f in 'index.html', 'check.html') { Copy-Item (Join-Path 'web' $f) $dist }
    foreach ($f in 'gps_web.js', 'gps_web_bg.wasm') { Copy-Item (Join-Path (Join-Path 'web' 'pkg') $f) $pkg }
    Get-ChildItem -Recurse -File $dist | ForEach-Object { '{0,10:N0}  {1}' -f $_.Length, $_.FullName.Substring($dist.Length + 1) }
} finally { Pop-Location }
