param(
    [string]$TaskName = "CryptoAiRunner"
)

$ErrorActionPreference = "Stop"
schtasks /Run /TN $TaskName | Out-Host
Write-Host "Task start requested: $TaskName"
