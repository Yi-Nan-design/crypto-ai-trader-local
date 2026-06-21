param(
    [string[]]$Symbols = @("ETHUSDT", "BNBUSDT"),
    [string]$Interval = "1h",
    [string]$RunnerInterval = "5m",
    [int]$LiveLimit = 800,
    [int]$ReviewEverySeconds = 900,
    [int]$Iterations = 0,
    [switch]$ExecuteOptimization,
    [switch]$ExecuteLiveTrain,
    [switch]$AllowLocalTraining,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CliArgs = @()
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$BundledPython = Join-Path $HOME ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"

foreach ($ConfigName in @("config.json", "config.default.json")) {
    $ConfigPath = Join-Path $Root $ConfigName
    if (Test-Path $ConfigPath) {
        $Config = Get-Content -LiteralPath $ConfigPath -Raw | ConvertFrom-Json
        if ($Config.live_trading_enabled -eq $true) {
            throw "$ConfigName has live_trading_enabled=true. Autonomous loop requires live_trading_enabled=false."
        }
    }
}

$Python = $BundledPython
if (Test-Path $VenvPython) {
    & $VenvPython -c "import numpy, pandas" 2>$null
    if ($LASTEXITCODE -eq 0) {
        $Python = $VenvPython
    }
}

function Normalize-Symbols {
    param([string[]]$Values)
    $Items = @()
    foreach ($Value in $Values) {
        if (-not $Value) {
            continue
        }
        $Items += $Value -split "," | ForEach-Object { $_.Trim() } | Where-Object { $_ }
    }
    return $Items
}

$SymbolArgs = Normalize-Symbols -Values $Symbols
$ArgsList = @(
    "-m", "crypto_ai_trader.cli",
    "autonomous-loop",
    "--symbols"
) + $SymbolArgs + @(
    "--interval", $Interval,
    "--runner-interval", $RunnerInterval,
    "--review-every-seconds", "$ReviewEverySeconds",
    "--iterations", "$Iterations"
)

if ($LiveLimit -gt 0) {
    $ArgsList += @("--live-limit", "$LiveLimit")
}
if ($ExecuteOptimization -or $AllowLocalTraining) {
    $ArgsList += "--execute-optimization"
}
if ($ExecuteLiveTrain) {
    $ArgsList += "--execute-live-train"
}
$ArgsList += $CliArgs

& $Python @ArgsList
