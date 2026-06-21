$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Candidates = @(
    "$env:LOCALAPPDATA\Programs\Microsoft VS Code\Code.exe",
    "$env:ProgramFiles\Microsoft VS Code\Code.exe",
    "${env:ProgramFiles(x86)}\Microsoft VS Code\Code.exe"
)

foreach ($Candidate in $Candidates) {
    if (Test-Path $Candidate) {
        Start-Process -FilePath $Candidate -ArgumentList @($Root)
        exit 0
    }
}

Write-Error "VS Code was not found. Install it first, then run this script again."
