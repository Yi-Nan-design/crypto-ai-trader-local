$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
& $Python -m crypto_ai_trader.runner stop
Write-Host "Runner stop requested."
