$ErrorActionPreference = 'Stop'
$archive = Join-Path $env:PREFIX 'computer-use.zip'
if (-not (Test-Path -LiteralPath $archive -PathType Leaf)) {
    throw 'The pinned HCU Windows archive has not been staged. Windows installation requires a published release archive.'
}
$expected = 'feb7c8699d2552fe2ae14da54485c9bca2ec8aecf77fc8f029b4a937dbbe1d6d'
$actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $archive).Hash.ToLower()
if ($actual -ne $expected) {
    throw "HCU Windows archive has sha256 $actual, not the pinned $expected"
}
Expand-Archive -Force -LiteralPath $archive -DestinationPath "$env:PREFIX\bin"
foreach ($executable in @('computer-use.exe', 'computer-use-windows-provider.exe', 'computer-use-input-ack.dll')) {
    if (-not (Test-Path -LiteralPath "$env:PREFIX\bin\$executable" -PathType Leaf)) {
        throw "HCU release is missing $executable"
    }
}
Remove-Item -LiteralPath $archive
