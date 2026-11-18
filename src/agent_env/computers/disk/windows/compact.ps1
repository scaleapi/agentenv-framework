# compact.ps1: run as agentadmin over ssh as a Windows disk is about to shut down to be packed, a base or a composed
# layer. It removes what Windows keeps that a computer doesn't need (update downloads, temp files, the delivery
# optimization cache), then trims the free space, which the launcher's disk, attached with discards, passes to its image,
# so packing it keeps only what Windows uses.
$ErrorActionPreference = 'Continue'
$ProgressPreference = 'SilentlyContinue'
$caches = @("$env:SystemRoot\SoftwareDistribution\Download", "$env:SystemRoot\Temp") +
    (Get-ChildItem C:\Users -Directory -Force | ForEach-Object { "$($_.FullName)\AppData\Local\Temp" })
foreach ($cache in $caches) { Get-ChildItem -Force $cache -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue }
Delete-DeliveryOptimizationCache -Force -ErrorAction SilentlyContinue
Optimize-Volume -DriveLetter C -ReTrim
