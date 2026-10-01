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
    # Stamp the build so the page loads a matching js/wasm pair (query-string cache busting) and shows its version.
    $sha = if ($env:GITHUB_SHA) { $env:GITHUB_SHA.Substring(0, 7) } else { (git rev-parse --short HEAD 2>$null) }
    $stamp = '{0}-{1}' -f (Get-Date -Format 'yyyyMMddHHmmss'), $sha
    $index = Join-Path $dist 'index.html'
    # explicit UTF-8 (no BOM) both ways: Windows PowerShell 5.1 would read the Japanese UI text as ANSI
    $utf8 = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($index, [System.IO.File]::ReadAllText($index, $utf8).Replace('__BUILD__', $stamp), $utf8)
    "build stamp: $stamp"
    foreach ($f in 'gps_web.js', 'gps_web_bg.wasm') { Copy-Item (Join-Path (Join-Path 'web' 'pkg') $f) $pkg }
    Get-ChildItem -Recurse -File $dist | ForEach-Object { '{0,10:N0}  {1}' -f $_.Length, $_.FullName.Substring($dist.Length + 1) }
} finally { Pop-Location }
