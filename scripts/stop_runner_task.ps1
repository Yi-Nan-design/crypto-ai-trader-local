param(
    [string]$TaskName = "CryptoAiRunner"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
& $Python -m crypto_ai_trader.runner stop
Start-Sleep -Seconds 5
schtasks /End /TN $TaskName | Out-Host
Write-Host "Task stop requested: $TaskName"
