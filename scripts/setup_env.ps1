$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$BundledPython = Join-Path $HOME ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"

if (!(Test-Path $VenvPython)) {
    & $BundledPython -m venv (Join-Path $Root ".venv")
}

& $VenvPython -m pip install --upgrade pip

$ProxyArgs = @()
$CommonProxyPorts = @(7890, 7891, 7897, 1080, 1087, 10808, 10809, 20170, 2080, 8080, 8888)
foreach ($Port in $CommonProxyPorts) {
    $Client = New-Object Net.Sockets.TcpClient
    try {
        $Connect = $Client.BeginConnect("127.0.0.1", $Port, $null, $null)
        if ($Connect.AsyncWaitHandle.WaitOne(200, $false)) {
            $Client.EndConnect($Connect)
            $ProxyArgs = @("--proxy", "http://127.0.0.1:$Port")
            break
        }
    } catch {
    } finally {
        $Client.Close()
    }
}

& $VenvPython -m pip install @ProxyArgs --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org -r (Join-Path $Root "requirements.txt")

Write-Host "Environment ready: $VenvPython"
