param(
    [string]$TaskName = "CryptoAiScheduledOptimizer"
)

$ErrorActionPreference = "Stop"
schtasks.exe /Query /TN $TaskName /V /FO LIST | Out-Host
