# agent-desktop's release, from the tarball the install downloaded and checked, in bin\ beside the driver, unpacked by
# the tar Windows ships. Its MSVC build loads VCRUNTIME140.dll, which Windows doesn't ship; the venv's Python, an MSVC
# build too, carries it, so it's copied beside the executable.
$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Force "$env:PREFIX\bin" | Out-Null
tar -xzf "$env:PREFIX\agent-desktop.tar.gz" -C "$env:PREFIX\bin"
if ($LASTEXITCODE) { throw "tar exited $LASTEXITCODE" }
Remove-Item "$env:PREFIX\agent-desktop.tar.gz"
$python = (Get-Content "$env:PREFIX\venv\pyvenv.cfg" | Where-Object { $_ -match '^home\s*=' }) -replace '^home\s*=\s*', ''
$runtime = Join-Path $python 'vcruntime140.dll'
if (-not (Test-Path $runtime)) { throw "no vcruntime140.dll in $python" }
Copy-Item $runtime "$env:PREFIX\bin"
