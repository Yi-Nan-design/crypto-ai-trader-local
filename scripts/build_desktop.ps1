param(
    [switch]$Installer
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
$OutputEncoding = [System.Text.UTF8Encoding]::new()

$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$NpmCache = Join-Path $Root ".cache\npm"
$ElectronCache = Join-Path $Root ".cache\electron"
$ElectronBuilderCache = Join-Path $Root ".cache\electron-builder"

if (!(Test-Path $Python)) {
    throw "Python venv not found: $Python. Run scripts\setup_env.ps1 before building the Electron desktop app."
}

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:ELECTRON_CACHE = $ElectronCache
$env:ELECTRON_BUILDER_CACHE = $ElectronBuilderCache

function Resolve-Tool {
    param(
        [string]$LocalPath,
        [string[]]$CommandNames
    )
    $Candidate = Join-Path $Root $LocalPath
    if (Test-Path $Candidate) {
        return $Candidate
    }
    foreach ($Name in $CommandNames) {
        $Command = Get-Command $Name -ErrorAction SilentlyContinue
        if ($Command) {
            return $Command.Source
        }
    }
    return $null
}

$Node = Resolve-Tool -LocalPath "tools\node\node.exe" -CommandNames @("node")
if (!$Node) {
    throw "Node.js was not found. Install Node.js or place a portable Node distribution at tools\node before building the Electron desktop app."
}

$Npm = Resolve-Tool -LocalPath "tools\node\npm.cmd" -CommandNames @("npm.cmd", "npm")
if (!$Npm) {
    throw "npm was not found. Install Node.js/npm or place a portable Node distribution at tools\node before building the Electron desktop app."
}

function Test-LocalProxy {
    try {
        $Client = [System.Net.Sockets.TcpClient]::new()
        $Connect = $Client.BeginConnect("127.0.0.1", 7890, $null, $null)
        if (!$Connect.AsyncWaitHandle.WaitOne(350)) {
            $Client.Close()
            return $false
        }
        $Client.EndConnect($Connect)
        $Client.Close()
        return $true
    }
    catch {
        return $false
    }
}

function Invoke-NpmChecked {
    param([string[]]$Arguments)
    & $Npm @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "npm failed with exit code ${LASTEXITCODE}: npm $($Arguments -join ' ')"
    }
}

if (!$env:HTTPS_PROXY -and (Test-LocalProxy)) {
    $env:HTTPS_PROXY = "http://127.0.0.1:7890"
    $env:HTTP_PROXY = "http://127.0.0.1:7890"
}

Push-Location $Root
try {
    New-Item -ItemType Directory -Force -Path $NpmCache | Out-Null
    New-Item -ItemType Directory -Force -Path $ElectronCache | Out-Null
    New-Item -ItemType Directory -Force -Path $ElectronBuilderCache | Out-Null
    if (Test-Path (Join-Path $Root "package-lock.json")) {
        Write-Host "Installing reproducible desktop dependencies with npm ci..."
        Invoke-NpmChecked -Arguments @("ci", "--cache", $NpmCache)
    }
    else {
        Write-Host "package-lock.json was not found. Running npm install to create one..."
        Invoke-NpmChecked -Arguments @("install", "--cache", $NpmCache)
    }

    if ($Installer) {
        Invoke-NpmChecked -Arguments @("run", "build:desktop:installer")
    }
    else {
        Invoke-NpmChecked -Arguments @("run", "build:desktop")
    }
}
finally {
    Pop-Location
}
