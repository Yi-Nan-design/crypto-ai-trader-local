param(
    [string[]]$Symbols = @("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT"),
    [string]$Interval = "1h",
    [string]$Start = "2024-01",
    [string]$End = "2025-12",
    [switch]$SkipDownload
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

$argsList = @("-m", "crypto_ai_trader.cli", "cycle", "--symbols") + $Symbols + @("--interval", $Interval, "--start", $Start, "--end", $End)
if ($SkipDownload) {
    $argsList += "--skip-download"
}

& $Python @argsList
