param(
    [switch]$SkipInstall,
    [switch]$SkipChecks
)

$ErrorActionPreference = "Stop"

function Test-ProjectRoot {
    param([string]$Path)

    return (
        (Test-Path -LiteralPath (Join-Path $Path "requirements.txt")) -and
        (Test-Path -LiteralPath (Join-Path $Path "crypto_ai_trader")) -and
        (Test-Path -LiteralPath (Join-Path $Path "scripts"))
    )
}

function Find-ProjectRoot {
    $Starts = @()
    if ($PSScriptRoot) {
        $Starts += (Resolve-Path -LiteralPath $PSScriptRoot).Path
    }
    $Starts += (Get-Location).Path

    foreach ($Start in $Starts) {
        $Item = Get-Item -LiteralPath $Start
        if (-not $Item.PSIsContainer) {
            $Item = $Item.Directory
        }

        while ($null -ne $Item) {
            if (Test-ProjectRoot -Path $Item.FullName) {
                return $Item.FullName
            }
            $Item = $Item.Parent
        }
    }

    throw "Could not locate the project root. Run this script from the extracted project or its scripts directory."
}

function Invoke-External {
    param(
        [string]$Label,
        [string]$File,
        [string[]]$Arguments
    )

    Write-Host ""
    Write-Host "==> $Label"
    & $File @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Label failed with exit code $LASTEXITCODE."
    }
}

function Test-PythonCommand {
    param(
        [string]$File,
        [string[]]$Arguments
    )

    & $File @Arguments "-c" "import sys; print(sys.version.split()[0])" 2>$null
    return ($LASTEXITCODE -eq 0)
}

function Resolve-BasePython {
    $PyLauncher = Get-Command "py" -ErrorAction SilentlyContinue
    if ($PyLauncher -and (Test-PythonCommand -File $PyLauncher.Source -Arguments @("-3"))) {
        return [PSCustomObject]@{
            File = $PyLauncher.Source
            Args = [string[]]@("-3")
            Display = "py -3"
        }
    }

    $Python = Get-Command "python" -ErrorAction SilentlyContinue
    if ($Python -and (Test-PythonCommand -File $Python.Source -Arguments @())) {
        return [PSCustomObject]@{
            File = $Python.Source
            Args = [string[]]@()
            Display = "python"
        }
    }

    throw "Python was not found. Install Python 3 and make sure 'py -3' or 'python' is available in PATH."
}

function Unblock-PowerShellFiles {
    param([string]$Root)

    if (-not (Get-Command "Unblock-File" -ErrorAction SilentlyContinue)) {
        Write-Warning "Unblock-File is not available in this PowerShell session; skipping unblock step."
        return
    }

    $SkippedDirs = @("\.git\", "\.venv\", "\node_modules\")
    $Count = 0
    $Files = Get-ChildItem -LiteralPath $Root -Recurse -File -Filter "*.ps1" -ErrorAction SilentlyContinue
    foreach ($File in $Files) {
        $FullName = $File.FullName
        $ShouldSkip = $false
        foreach ($Dir in $SkippedDirs) {
            if ($FullName.Contains($Dir)) {
                $ShouldSkip = $true
                break
            }
        }
        if ($ShouldSkip) {
            continue
        }

        try {
            Unblock-File -LiteralPath $FullName -ErrorAction Stop
            $Count += 1
        }
        catch {
            Write-Warning "Could not unblock ${FullName}: $($_.Exception.Message)"
        }
    }

    Write-Host "Unblocked $Count PowerShell script(s)."
}

$Root = Find-ProjectRoot
$VenvDir = Join-Path $Root ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$Requirements = Join-Path $Root "requirements.txt"

Write-Host "Project root: $Root"
Set-Location -LiteralPath $Root

Unblock-PowerShellFiles -Root $Root

if (-not (Test-Path -LiteralPath $VenvPython)) {
    $BasePython = Resolve-BasePython
    Write-Host "Creating virtual environment with $($BasePython.Display)."
    $VenvArgs = @()
    $VenvArgs += $BasePython.Args
    $VenvArgs += @("-m", "venv", $VenvDir)
    Invoke-External -Label "Create .venv" -File $BasePython.File -Arguments $VenvArgs
}
else {
    Write-Host "Using existing virtual environment: $VenvPython"
}

if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "Virtual environment Python was not found at $VenvPython."
}

if (-not $SkipInstall) {
    Invoke-External -Label "Upgrade pip" -File $VenvPython -Arguments @("-m", "pip", "install", "--upgrade", "pip")
    Invoke-External -Label "Install requirements.txt" -File $VenvPython -Arguments @("-m", "pip", "install", "-r", $Requirements)
}
else {
    Write-Host "Skipping dependency install because -SkipInstall was provided."
}

if (-not $SkipChecks) {
    Invoke-External -Label "Run smoke check" -File $VenvPython -Arguments @("-m", "crypto_ai_trader.cli", "smoke")
    Invoke-External -Label "Run doctor check" -File $VenvPython -Arguments @("-m", "crypto_ai_trader.cli", "doctor")
}
else {
    Write-Host "Skipping smoke and doctor because -SkipChecks was provided."
}

Write-Host ""
Write-Host "Portable restore complete."
Write-Host "Scheduled tasks are machine-local and must be reinstalled on this Windows user account."
Write-Host "Do not copy old scheduled tasks from another machine or path."
Write-Host "To reinstall the runner task, run:"
Write-Host 'powershell -ExecutionPolicy Bypass -File ".\scripts\install_runner_task.ps1"'
