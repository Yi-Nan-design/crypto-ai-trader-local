param(
    [string[]]$Symbols = @("ETHUSDT"),
    [string]$Interval = "5m",
    [int]$Limit = 500,
    [int]$Iterations = 1,
    [int]$SleepSeconds = 60
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$BundledPython = Join-Path $HOME ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$Python = $BundledPython
if (Test-Path $VenvPython) {
    & $VenvPython -c "import numpy, pandas" 2>$null
    if ($LASTEXITCODE -eq 0) {
        $Python = $VenvPython
    }
}

$argsList = @(
    "-m", "crypto_ai_trader.cli",
    "live-train",
    "--symbols"
) + $Symbols + @(
    "--interval", $Interval,
    "--limit", "$Limit",
    "--iterations", "$Iterations",
    "--sleep-seconds", "$SleepSeconds"
)

& $Python @argsList
