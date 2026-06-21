param(
    [string]$TaskName = "CryptoAiRunner"
)

$ErrorActionPreference = "Stop"
schtasks /Query /TN $TaskName /V /FO LIST
