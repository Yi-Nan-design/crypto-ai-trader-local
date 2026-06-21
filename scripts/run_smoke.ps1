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

& $Python -m crypto_ai_trader.cli smoke
