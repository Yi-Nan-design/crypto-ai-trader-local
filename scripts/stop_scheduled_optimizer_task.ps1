param(
    [string]$TaskName = "CryptoAiScheduledOptimizer"
)

$ErrorActionPreference = "Stop"
$PreviousPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    $Output = & schtasks.exe /End /TN $TaskName 2>&1
    $ExitCode = $LASTEXITCODE
}
finally {
    $ErrorActionPreference = $PreviousPreference
}

if ($Output) {
    $Output | Out-Host
}

if ($ExitCode -ne 0) {
    Write-Host "Task may not be running: $TaskName"
}
else {
    Write-Host "Scheduled optimizer stop requested: $TaskName"
}
