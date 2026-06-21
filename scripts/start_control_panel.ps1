$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"
$BundledPython = Join-Path $HOME ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$Python = if (Test-Path $VenvPython) { $VenvPython } else { $BundledPython }

function Test-PortFree {
    param([int]$Port)
    $Listener = $null
    try {
        $Address = [System.Net.IPAddress]::Parse("127.0.0.1")
        $Listener = [System.Net.Sockets.TcpListener]::new($Address, $Port)
        $Listener.Start()
        return $true
    }
    catch {
        return $false
    }
    finally {
        if ($Listener) {
            $Listener.Stop()
        }
    }
}

$Port = $null
foreach ($Candidate in 8765..8775) {
    if (Test-PortFree -Port $Candidate) {
        $Port = $Candidate
        break
    }
}

if (-not $Port) {
    throw "No free local port found between 8765 and 8775."
}

$Url = "http://127.0.0.1:$Port"
Write-Host "Control panel: $Url"
Start-Process $Url
& $Python -m crypto_ai_trader.dashboard_server --host 127.0.0.1 --port $Port
