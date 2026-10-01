# Builds a static distribution in dist/ (copy it to any HTTPS static host).
# Contains only what the page loads: index.html, check.html, pkg/gps_web.js, pkg/gps_web_bg.wasm.
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    & "$PSScriptRoot\build-web.ps1"
    $dist = Join-Path $PSScriptRoot 'dist'
    if (Test-Path $dist) { Remove-Item -Recurse -Force $dist }
    New-Item -ItemType Directory -Force "$dist\pkg" | Out-Null
    Copy-Item web\index.html, web\check.html $dist
    Copy-Item web\pkg\gps_web.js, web\pkg\gps_web_bg.wasm "$dist\pkg"
    Get-ChildItem -Recurse $dist | Where-Object { -not $_.PSIsContainer } |
        ForEach-Object { '{0,10:N0}  {1}' -f $_.Length, $_.FullName.Substring($dist.Length + 1) }
} finally { Pop-Location }
