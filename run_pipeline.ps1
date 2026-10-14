$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$previousLocation = Get-Location

try {
    Set-Location -LiteralPath $projectRoot
    $env:PYTHONPATH = Join-Path $projectRoot "src"

    foreach ($module in @(
        "stocks_forecast.ingest",
        "stocks_forecast.prepare",
        "stocks_forecast.modeling",
        "stocks_forecast.feature_analysis"
    )) {
        Write-Host "Running $module"
        & python -m $module
        if ($LASTEXITCODE -ne 0) {
            throw "$module failed with exit code $LASTEXITCODE"
        }
    }
}
finally {
    Set-Location -LiteralPath $previousLocation
}
