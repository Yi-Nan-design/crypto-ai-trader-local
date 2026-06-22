param(
    [string[]]$Symbols = @("ETHUSDT", "BNBUSDT"),
    [string]$Interval = "5m",
    [int]$Limit = 800,
    [int]$TrainEverySeconds = 900,
    [int]$MaxModelTrials = 4,
    [double]$TimeBudgetMinutes = 6,
    [int]$RollingFolds = 1,
    [int]$MaxTrainingRows = 8000
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$LogDir = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Symbols = @(
    $Symbols |
        ForEach-Object { $_ -split "," } |
        ForEach-Object { $_.Trim().ToUpperInvariant() } |
        Where-Object { $_ }
)

$ArgsList = @(
    "-m", "crypto_ai_trader.runner",
    "run",
    "--symbols"
) + $Symbols + @(
    "--interval", $Interval,
    "--limit", "$Limit",
    "--train-every-seconds", "$TrainEverySeconds",
    "--live-max-model-trials", "$MaxModelTrials",
    "--live-time-budget-minutes", "$TimeBudgetMinutes",
    "--live-rolling-folds", "$RollingFolds",
    "--live-max-training-rows", "$MaxTrainingRows"
)

$OutLog = Join-Path $LogDir "runner.out.log"
$ErrLog = Join-Path $LogDir "runner.err.log"

$QuotedArgs = ($ArgsList | ForEach-Object {
    if ($_ -match '[\s"]') {
        '"' + ($_ -replace '"', '\"') + '"'
    } else {
        $_
    }
}) -join " "

$Command = 'cd /d "{0}" && "{1}" {2} >> "{3}" 2>> "{4}"' -f $Root, $Python, $QuotedArgs, $OutLog, $ErrLog
$ProcessInfo = New-Object System.Diagnostics.ProcessStartInfo
$ProcessInfo.FileName = "cmd.exe"
$ProcessInfo.Arguments = '/c "' + $Command + '"'
$ProcessInfo.WorkingDirectory = $Root
$ProcessInfo.UseShellExecute = $false
$ProcessInfo.CreateNoWindow = $true

$DuplicatePathKeys = @()
foreach ($Key in $ProcessInfo.EnvironmentVariables.Keys) {
    if ($Key -ceq "PATH") {
        $DuplicatePathKeys += $Key
    }
}
foreach ($Key in $DuplicatePathKeys) {
    $ProcessInfo.EnvironmentVariables.Remove($Key)
}
$ProcessInfo.EnvironmentVariables["Path"] = $env:Path

$Process = [System.Diagnostics.Process]::Start($ProcessInfo)

Write-Host "Runner started with launcher PID $($Process.Id). Check status with scripts\status_runner.ps1"
