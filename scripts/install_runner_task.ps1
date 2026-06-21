param(
    [string[]]$Symbols = @("ETHUSDT", "BNBUSDT"),
    [string]$Interval = "5m",
    [int]$Limit = 800,
    [int]$TrainEverySeconds = 900,
    [string]$TaskName = "CryptoAiRunner"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$LogDir = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$SymbolArgs = ($Symbols -join " ")
$OutLog = Join-Path $LogDir "runner.task.out.log"
$ErrLog = Join-Path $LogDir "runner.task.err.log"
$LauncherPath = Join-Path $env:USERPROFILE "crypto_ai_runner.ps1"

$EscapedRoot = $Root.Replace("'", "''")
$EscapedPython = $Python.Replace("'", "''")
$EscapedOutLog = $OutLog.Replace("'", "''")
$EscapedErrLog = $ErrLog.Replace("'", "''")
@"
`$ErrorActionPreference = "Stop"
try {
    Set-Location -LiteralPath '$EscapedRoot'
    & '$EscapedPython' -m crypto_ai_trader.runner run --symbols $SymbolArgs --interval $Interval --limit $Limit --train-every-seconds $TrainEverySeconds >> '$EscapedOutLog' 2>> '$EscapedErrLog'
    exit `$LASTEXITCODE
}
catch {
    `$Message = "[{0}] {1}" -f (Get-Date -Format "s"), `$_.Exception.Message
    Add-Content -LiteralPath '$EscapedErrLog' -Value `$Message
    exit 1
}
"@ | Set-Content -Path $LauncherPath -Encoding UTF8

$Command = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$LauncherPath`""
$StartupShortcut = Join-Path ([Environment]::GetFolderPath("Startup")) "CryptoAiRunner.lnk"

function Invoke-ScheduledTaskCommand {
    param(
        [string[]]$Arguments,
        [switch]$AllowFailure
    )

    $PreviousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $Output = & schtasks.exe @Arguments 2>&1
        $ExitCode = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $PreviousPreference
    }

    if ($Output -and (-not $AllowFailure -or $ExitCode -eq 0)) {
        $Output | Out-Host
    }

    if ($ExitCode -ne 0 -and -not $AllowFailure) {
        throw "schtasks.exe failed with exit code ${ExitCode}: $($Arguments -join ' ')"
    }

    return $ExitCode
}

$QueryExitCode = Invoke-ScheduledTaskCommand -Arguments @("/Query", "/TN", $TaskName) -AllowFailure
if ($QueryExitCode -eq 0) {
    Invoke-ScheduledTaskCommand -Arguments @("/Delete", "/TN", $TaskName, "/F") | Out-Null
}

Invoke-ScheduledTaskCommand -Arguments @("/Create", "/TN", $TaskName, "/TR", $Command, "/SC", "ONCE", "/ST", "23:59", "/F") | Out-Null

$Shell = New-Object -ComObject WScript.Shell
$Shortcut = $Shell.CreateShortcut($StartupShortcut)
$Shortcut.TargetPath = "powershell.exe"
$Shortcut.Arguments = "-NoProfile -ExecutionPolicy Bypass -WindowStyle Minimized -File `"$LauncherPath`""
$Shortcut.WorkingDirectory = $Root
$Shortcut.WindowStyle = 7
$Shortcut.Description = "Start local crypto AI runner"
$Shortcut.Save()

Write-Host "Installed task: $TaskName"
Write-Host "Launcher: $LauncherPath"
Write-Host "Startup shortcut: $StartupShortcut"
Write-Host "Start it with scripts\start_runner_task.ps1"
