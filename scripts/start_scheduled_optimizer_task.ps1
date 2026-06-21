param(
    [string]$TaskName = "CryptoAiScheduledOptimizer"
)

$ErrorActionPreference = "Stop"
schtasks.exe /Run /TN $TaskName | Out-Host
Write-Host "Scheduled optimizer start requested: $TaskName"
