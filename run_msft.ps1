param(
    [switch]$Refresh,
    [switch]$PrepareOnly
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonPath = Join-Path $projectRoot '.venv/Scripts/python.exe'
$previousLocation = Get-Location
try {
    Set-Location -LiteralPath $projectRoot
    if (-not (Test-Path -LiteralPath $pythonPath)) {
        throw 'Create .venv and install requirements-dashboard.lock.txt first; see README.'
    }
    & $pythonPath scripts/explain_msft.py
    if ($LASTEXITCODE -ne 0) { throw 'Explanation preparation failed. Prepare the frozen experiment first.' }
    $updateArgs = @('scripts/update_msft.py')
    if ($Refresh) { $updateArgs += '--refresh' }
    & $pythonPath @updateArgs
    if ($LASTEXITCODE -ne 0) { throw 'Operational update failed; previous result retained.' }
    if (-not $PrepareOnly) {
        & $pythonPath -m streamlit run scripts/dashboard_msft.py --server.address 127.0.0.1 --server.port 8501 --server.headless true --browser.gatherUsageStats false
        if ($LASTEXITCODE -ne 0) { throw 'Dashboard server exited with an error.' }
    }
}
finally {
    Set-Location -LiteralPath $previousLocation
}
