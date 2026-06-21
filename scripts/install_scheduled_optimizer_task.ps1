param(
    [string[]]$Symbols = @("SOLUSDT", "ETHUSDT", "BNBUSDT"),
    [string[]]$Intervals = @("1h"),
    [double]$TimeBudgetMinutes = 12,
    [int]$MaxModelTrials = 1,
    [int]$MaxTrainingRows = 12000,
    [string]$Complexity = "standard",
    [int]$RollingFolds = 0,
    [int]$MaxTargets = 1,
    [string]$TaskName = "CryptoAiScheduledOptimizer"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$LogDir = Join-Path $Root "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python virtual environment not found: $Python. Run scripts\setup_env.ps1 first."
}

$AllowedComplexity = @("standard", "expanded", "deep", "blackbox")
if ($AllowedComplexity -notcontains $Complexity) {
    $Complexity = "standard"
}

$SymbolArgs = ($Symbols | ForEach-Object { $_.Trim().ToUpperInvariant() } | Where-Object { $_ }) -join " "
$IntervalArgs = ($Intervals | ForEach-Object { $_.Trim() } | Where-Object { $_ }) -join " "
$SymbolValues = @($Symbols | ForEach-Object { $_.Trim().ToUpperInvariant() } | Where-Object { $_ })
$IntervalValues = @($Intervals | ForEach-Object { $_.Trim() } | Where-Object { $_ })
function ConvertTo-ArrayLiteral {
    param([string[]]$Values)
    return (($Values | ForEach-Object { "'$($_.Replace("'", "''"))'" }) -join ", ")
}
$SymbolArrayLiteral = ConvertTo-ArrayLiteral -Values $SymbolValues
$IntervalArrayLiteral = ConvertTo-ArrayLiteral -Values $IntervalValues
$OutLog = Join-Path $LogDir "scheduled_optimizer.task.out.log"
$ErrLog = Join-Path $LogDir "scheduled_optimizer.task.err.log"
$LauncherPath = Join-Path $env:USERPROFILE "crypto_ai_scheduled_optimizer.ps1"

$EscapedRoot = $Root.Replace("'", "''")
$EscapedPython = $Python.Replace("'", "''")
$EscapedOutLog = $OutLog.Replace("'", "''")
$EscapedErrLog = $ErrLog.Replace("'", "''")
@"
`$ErrorActionPreference = "Stop"
try {
    Set-Location -LiteralPath '$EscapedRoot'
    `$RunId = Get-Date -Format "yyyyMMddHHmmss"
    `$TempOut = '$EscapedOutLog' + "." + `$RunId + ".tmp"
    `$TempErr = '$EscapedErrLog' + "." + `$RunId + ".tmp"
    `$Args = @(
        "-m", "crypto_ai_trader.cli", "scheduled-optimize",
        "--symbols"
    ) + @($SymbolArrayLiteral) + @(
        "--intervals"
    ) + @($IntervalArrayLiteral) + @(
        "--include-realtime",
        "--complexity", "$Complexity",
        "--rolling-folds", "$RollingFolds",
        "--time-budget-minutes", "$TimeBudgetMinutes",
        "--max-model-trials", "$MaxModelTrials",
        "--max-training-rows", "$MaxTrainingRows",
        "--max-targets", "$MaxTargets"
    )
    `$Process = Start-Process -FilePath '$EscapedPython' -ArgumentList `$Args -WorkingDirectory '$EscapedRoot' -RedirectStandardOutput `$TempOut -RedirectStandardError `$TempErr -WindowStyle Hidden -PassThru
    `$TimeoutSeconds = [int][Math]::Ceiling(($TimeBudgetMinutes + 3) * 60)
    if (-not `$Process.WaitForExit(`$TimeoutSeconds * 1000)) {
        `$TimeoutMessage = "[{0}] scheduled-optimize exceeded hard timeout of {1}s; terminating process tree PID={2}" -f (Get-Date -Format "s"), `$TimeoutSeconds, `$Process.Id
        Add-Content -LiteralPath '$EscapedErrLog' -Value `$TimeoutMessage
        & taskkill.exe /PID `$Process.Id /T /F >> '$EscapedErrLog' 2>&1
        `$ExitCode = 124
    }
    else {
        `$ExitCode = `$Process.ExitCode
    }
    if (Test-Path -LiteralPath `$TempOut) {
        Get-Content -LiteralPath `$TempOut | Add-Content -LiteralPath '$EscapedOutLog'
        Remove-Item -LiteralPath `$TempOut -Force
    }
    if (Test-Path -LiteralPath `$TempErr) {
        Get-Content -LiteralPath `$TempErr | Add-Content -LiteralPath '$EscapedErrLog'
        Remove-Item -LiteralPath `$TempErr -Force
    }
    exit `$ExitCode
}
catch {
    `$Message = "[{0}] {1}" -f (Get-Date -Format "s"), `$_.Exception.Message
    Add-Content -LiteralPath '$EscapedErrLog' -Value `$Message
    exit 1
}
"@ | Set-Content -Path $LauncherPath -Encoding UTF8

$Command = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$LauncherPath`""

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

Invoke-ScheduledTaskCommand -Arguments @("/Create", "/TN", $TaskName, "/TR", $Command, "/SC", "HOURLY", "/MO", "1", "/F") | Out-Null

try {
    $Task = Get-ScheduledTask -TaskName $TaskName -ErrorAction Stop
    $Task.Settings.DisallowStartIfOnBatteries = $false
    $Task.Settings.StopIfGoingOnBatteries = $false
    $Task.Settings.StartWhenAvailable = $true
    Set-ScheduledTask -TaskName $TaskName -Settings $Task.Settings | Out-Null
}
catch {
    Write-Warning "Task was created, but power-condition settings could not be updated: $($_.Exception.Message)"
}

Write-Host "Installed hourly optimizer task: $TaskName"
Write-Host "Launcher: $LauncherPath"
Write-Host "stdout: $OutLog"
Write-Host "stderr: $ErrLog"
Write-Host "Start now with scripts\start_scheduled_optimizer_task.ps1"
