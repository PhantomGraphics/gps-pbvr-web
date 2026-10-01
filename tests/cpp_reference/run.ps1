# Compiles dump_oracle.cpp against the repo's C++ oracle with MSVC and regenerates
# tests/fixtures/cpp_*.f64 (raw f64 RGB). Needs a VS2026 install (vswhere).
$ErrorActionPreference = 'Stop'
$here = $PSScriptRoot; $root = (Resolve-Path "$here\..\..\..\..").Path
$vs = & "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property installationPath
$out = (Resolve-Path "$here\..\fixtures").Path.Replace('\', '/')
$obj = Join-Path $env:TEMP 'gps_dump_oracle'; New-Item -ItemType Directory -Force $obj | Out-Null
$gsv = "$root\Phantom\PointCloud\GSView"; $glm = "$root\Phantom\CGLib\ThirdParty\glm-0.9.9.8"
$bat = "$obj\build.bat"
@"
@echo off
call "$vs\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1
cl /nologo /std:c++17 /O2 /EHsc /utf-8 "/DOUT_DIR=\"$out\"" /I"$gsv" /I"$glm" "$here\dump_oracle.cpp" "$gsv\GaussianPointOracle.cpp" "$gsv\GaussianPointMath.cpp" /Fo"$obj\\" /Fe"$obj\dump_oracle.exe" || exit /b 1
"$obj\dump_oracle.exe"
"@ | Set-Content -Encoding ascii $bat
cmd /c $bat
if ($LASTEXITCODE) { throw 'C++ reference dump failed' }
