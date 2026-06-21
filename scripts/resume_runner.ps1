$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
& $Python -m crypto_ai_trader.runner resume
Write-Host "Runner resume requested."
