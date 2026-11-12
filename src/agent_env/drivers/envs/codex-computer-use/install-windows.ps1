# The pinned ChatGPT app's resources, from the MSIX the install downloaded and checked, unpacked beside the driver with
# its entries' names decoded, and never run, so it never updates itself. OpenAI serves only its latest MSIX, so the
# pin fails the next install once that moves on, and the pin is bumped.
$ErrorActionPreference = 'Stop'
& "$env:PREFIX\venv\Scripts\python.exe" "$env:PREFIX\unpack-msix.py" "$env:PREFIX\ChatGPT.msix" "$env:PREFIX\ChatGPT\resources"
if ($LASTEXITCODE) { throw "unpack-msix.py exited $LASTEXITCODE" }
Remove-Item "$env:PREFIX\ChatGPT.msix"
