$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$BundledPython = Join-Path $HOME ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$Python = if (Test-Path $VenvPython) { $VenvPython } else { $BundledPython }

& $Python -m crypto_ai_trader.dashboard_server --host 127.0.0.1 --port 8765
