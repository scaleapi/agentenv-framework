# Run elevated once the app's files are in: whether it is, in the folder it was given.
$identity = [Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()
$elevated = $identity.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
"elevated $elevated in $env:PREFIX" | Set-Content -Encoding ascii run.txt
