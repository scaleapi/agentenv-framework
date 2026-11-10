# setup.ps1: a Windows base readied for agent-env, once, elevated, at agentadmin's first sign-in. OpenSSH comes first
# (agentadmin by agent-env's key alone), which is how agent-env composes the base and reads this setup's log when it
# fails (setup-failed). Then nothing of Windows' own is left in an agent's way: no updates, lock screen, notifications,
# widgets, Copilot, OneDrive, first-run pages or apps Windows installs in the background, and no page file, all of
# which would grow every layer composed on the base. The pinned Python, with Tk, is what layers' venvs and the probe's
# fixture run on; agentenv-session starts the session's services, windowless, at the session user's sign-in. Then the
# session user signs in at every boot.
param([string]$Media, [string]$User, [string]$UserPassword)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$root = 'C:\agentenv'
foreach ($folder in 'bin', 'logs', 'services', 'drivers', 'apps') { New-Item -ItemType Directory -Force "$root\$folder" | Out-Null }
Start-Transcript -Path "$root\logs\setup.log" -Append

# The network may come up after the first sign-in starts, so a fetch is tried for five minutes.
function Fetch([string]$Url, [string]$Sha256, [string]$To) {
    for ($try = 1; ; $try++) {
        try { Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $To; break }
        catch { if ($try -ge 30) { throw } ; Start-Sleep 10 }
    }
    $got = (Get-FileHash -Algorithm SHA256 $To).Hash.ToLower()
    if ($got -ne $Sha256) { throw "setup: $Url came with sha256 $got, not $Sha256" }
}

# reg.exe makes a key's missing parents and keeps its other values. A cosmetic value Windows may protect, as it does
# some of a user's Explorer settings, is set with -BestEffort, which logs a refusal rather than failing setup.
function Set-Values([string]$Key, [hashtable]$Values, [switch]$BestEffort) {
    foreach ($name in $Values.Keys) {
        reg.exe add $Key /v $name /t REG_DWORD /d $Values[$name] /f | Out-Null
        if ($LASTEXITCODE -and $BestEffort) { "setup: Windows kept $Key\$name as it was" }
        elseif ($LASTEXITCODE) { throw "setup: reg add $Key /v $name exited $LASTEXITCODE" }
    }
}

try {
    Fetch 'https://github.com/PowerShell/Win32-OpenSSH/releases/download/10.4.0.0p1-Preview/OpenSSH-Win64-v10.4.0.0.msi' `
        '788d8643464a01eb63640a2fd746f9ed23d106b5a901a35f88dc8cd40e446ac9' "$env:TEMP\openssh.msi"
    Start-Process msiexec.exe -Wait -ArgumentList "/i $env:TEMP\openssh.msi /qn"
    New-Item -ItemType Directory -Force C:\ProgramData\ssh | Out-Null
    Copy-Item (Join-Path $Media 'authorized_keys') C:\ProgramData\ssh\administrators_authorized_keys
    icacls C:\ProgramData\ssh\administrators_authorized_keys /inheritance:r /grant '*S-1-5-32-544:F' /grant 'SYSTEM:F' | Out-Null
    # QEMU's user network reaches the guest at the ports the launcher forwards alone, which the layers' services listen on.
    Set-NetFirewallProfile -Profile Domain, Private, Public -Enabled False
    # The launcher, at 10.0.2.2, forwards a local grant server's port to the machine it runs on, as host.docker.internal.
    Add-Content -Path "$env:SystemRoot\System32\drivers\etc\hosts" -Value "`r`n10.0.2.2 host.docker.internal" -Encoding ASCII
    Set-Service sshd -StartupType Automatic
    Start-Service sshd

    Set-Values 'HKLM\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate\AU' @{ NoAutoUpdate = 1 }
    Set-Values 'HKLM\SOFTWARE\Policies\Microsoft\Windows\Personalization' @{ NoLockScreen = 1 }
    Set-Values 'HKLM\SOFTWARE\Policies\Microsoft\Windows\CloudContent' @{ DisableWindowsConsumerFeatures = 1; DisableSoftLanding = 1; DisableCloudOptimizedContent = 1 }
    Set-Values 'HKLM\SOFTWARE\Policies\Microsoft\Dsh' @{ AllowNewsAndInterests = 0 }
    Set-Values 'HKLM\SOFTWARE\Policies\Microsoft\Windows\WindowsCopilot' @{ TurnOffWindowsCopilot = 1 }
    Set-Values 'HKLM\SOFTWARE\Policies\Microsoft\Windows\OOBE' @{ DisablePrivacyExperience = 1 }
    Set-Values 'HKLM\SOFTWARE\Policies\Microsoft\Windows\Explorer' @{ DisableNotificationCenter = 1; HideRecommendedSection = 1 }
    Set-Values 'HKLM\SOFTWARE\Policies\Microsoft\Windows\Windows Search' @{ EnableDynamicContentInWSB = 0 }
    Set-Values 'HKLM\SOFTWARE\Policies\Microsoft\Edge' @{ HideFirstRunExperience = 1 }
    Set-Values 'HKLM\SOFTWARE\Policies\Microsoft\WindowsStore' @{ AutoDownload = 2 }
    Set-Values 'HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Policies\System' @{ EnableFirstLogonAnimation = 0 }
    Set-Values 'HKLM\SYSTEM\CurrentControlSet\Control\BitLocker' @{ PreventDeviceEncryption = 1 }
    Set-Values 'HKLM\SYSTEM\CurrentControlSet\Control\CI\Policy' @{ VerifiedAndReputablePolicyState = 0 }

    reg.exe load HKU\AgentenvDefault C:\Users\Default\NTUSER.DAT | Out-Null
    Set-Values 'HKU\AgentenvDefault\Software\Microsoft\Windows\CurrentVersion\PushNotifications' @{ ToastEnabled = 0 } -BestEffort
    Set-Values 'HKU\AgentenvDefault\Software\Microsoft\Windows\CurrentVersion\Explorer\Advanced' @{ ShowCopilotButton = 0; Start_IrisRecommendations = 0 } -BestEffort
    Set-Values 'HKU\AgentenvDefault\Software\Microsoft\Windows\CurrentVersion\UserProfileEngagement' @{ ScoobeSystemSettingEnabled = 0 } -BestEffort
    Set-Values 'HKU\AgentenvDefault\Software\Microsoft\Windows\CurrentVersion\ContentDeliveryManager' @{
        SubscribedContentEnabled = 0; SystemPaneSuggestionsEnabled = 0; SilentInstalledAppsEnabled = 0; SoftLandingEnabled = 0 } -BestEffort
    # OneDrive sets itself up at a new user's first sign-in, from the default profile's Run key, and leaves two RunOnce
    # deletes behind. Each is a console command, which Windows opens in a Terminal window that takes the foreground, and
    # a composed disk boots afresh every time, so they would run at every boot and take what an agent types.
    $run = 'Registry::HKU\AgentenvDefault\Software\Microsoft\Windows\CurrentVersion\Run'
    (Get-Item $run -ErrorAction SilentlyContinue).Property |
        Where-Object { (Get-ItemPropertyValue $run $_) -match 'OneDriveSetup' } | ForEach-Object { Remove-ItemProperty $run $_ }
    [gc]::Collect()
    reg.exe unload HKU\AgentenvDefault | Out-Null

    # Windows' background installs: Edge's updater, which brings the Copilot app, and the new Outlook and Dev Home, which
    # Windows installs after a first sign-in. Each would land in every composed layer and run at every boot.
    Set-Values 'HKLM\SOFTWARE\Policies\Microsoft\EdgeUpdate' @{ UpdateDefault = 0; InstallDefault = 0; AutoUpdateCheckPeriodMinutes = 0 }
    foreach ($service in 'edgeupdate', 'edgeupdatem') {
        Stop-Service $service -Force -ErrorAction SilentlyContinue
        Set-Service $service -StartupType Disabled -ErrorAction SilentlyContinue
    }
    Get-ScheduledTask -TaskName 'MicrosoftEdgeUpdate*' -ErrorAction SilentlyContinue | Disable-ScheduledTask | Out-Null
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue `
        'HKLM:\SOFTWARE\Microsoft\WindowsUpdate\Orchestrator\UScheduler_Oobe\OutlookUpdate',
        'HKLM:\SOFTWARE\Microsoft\WindowsUpdate\Orchestrator\UScheduler_Oobe\DevHomeUpdate'

    powercfg /change standby-timeout-ac 0
    powercfg /change monitor-timeout-ac 0
    powercfg /hibernate off
    $system = Get-CimInstance Win32_ComputerSystem
    Set-CimInstance -InputObject $system -Property @{ AutomaticManagedPagefile = $false }
    Get-CimInstance Win32_PageFileSetting | Remove-CimInstance

    Fetch 'https://github.com/astral-sh/python-build-standalone/releases/download/20261026/cpython-3.12.16%2B20261026-x86_64-pc-windows-msvc-install_only.tar.gz' `
        '590dfcbe3a34c07301e6332e13e2188b98655cd072b08afa45a7998dc4c22bc5' "$env:TEMP\python.tar.gz"
    tar -xzf "$env:TEMP\python.tar.gz" -C $root
    if ($LASTEXITCODE) { throw "setup: unpacking the pinned Python exited $LASTEXITCODE" }
    Copy-Item "$root\python\python.exe" "$root\python\python3.exe"
    & "$root\python\python.exe" -c 'import tkinter; assert tkinter.TkVersion >= 8.6'
    if ($LASTEXITCODE) { throw 'setup: the pinned Python has no Tk' }
    $path = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    [Environment]::SetEnvironmentVariable('Path', "$root\bin;$root\python;$root\python\Scripts;$path", 'Machine')

    Copy-Item (Join-Path $Media 'agentenv-session.py') "$root\bin\agentenv-session.py"
    $action = New-ScheduledTaskAction -Execute "$root\python\pythonw.exe" -Argument "$root\bin\agentenv-session.py"
    $principal = New-ScheduledTaskPrincipal -GroupId 'S-1-5-32-545' -RunLevel Limited
    $settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit 0 -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    Register-ScheduledTask -TaskName agentenv-session -Action $action -Trigger (New-ScheduledTaskTrigger -AtLogOn) `
        -Principal $principal -Settings $settings | Out-Null

    $winlogon = 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Winlogon'
    Set-ItemProperty $winlogon AutoAdminLogon '1'
    Set-ItemProperty $winlogon DefaultUserName $User
    Set-ItemProperty $winlogon DefaultDomainName '.'
    Set-ItemProperty $winlogon DefaultPassword $UserPassword
    Remove-ItemProperty $winlogon AutoLogonCount -ErrorAction SilentlyContinue
} catch {
    "$_`n$($_.ScriptStackTrace)" | Out-File "$root\setup-failed"
    Stop-Transcript
    exit 1
}
'ready' | Out-File "$root\setup-done"
Stop-Transcript
shutdown /r /t 5
