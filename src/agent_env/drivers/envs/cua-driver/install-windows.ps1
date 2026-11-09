# The cua-driver release's zip, fetched beside the driver, unpacked into its cua-driver folder.
$ErrorActionPreference = 'Stop'
Expand-Archive -Force -Path "$env:PREFIX\cua-driver.zip" -DestinationPath "$env:PREFIX\unpacked"
Move-Item "$env:PREFIX\unpacked\cua-driver-rs-0.38.0-windows-x86_64" "$env:PREFIX\cua-driver"
Remove-Item -Recurse "$env:PREFIX\unpacked", "$env:PREFIX\cua-driver.zip"
