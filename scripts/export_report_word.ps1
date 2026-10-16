param(
    [Parameter(Mandatory=$true)][string]$InputDocx,
    [Parameter(Mandatory=$true)][string]$OutputPdf
)
$ErrorActionPreference = 'Stop'
$inputPath = (Resolve-Path -LiteralPath $InputDocx).Path
$pdfPath = [System.IO.Path]::GetFullPath($OutputPdf)
New-Item -ItemType Directory -Force ([System.IO.Path]::GetDirectoryName($pdfPath)) | Out-Null
$wordApp = New-Object -ComObject Word.Application
try {
    $wordApp.Visible = $false
    $wordApp.DisplayAlerts = 0
    $reportDoc = $wordApp.Documents.Open($inputPath, $false, $false)
    try {
        $reportDoc.Fields.Update() | Out-Null
        foreach ($toc in $reportDoc.TablesOfContents) { $toc.Update() }
        $reportDoc.Repaginate()
        foreach ($toc in $reportDoc.TablesOfContents) { $toc.UpdatePageNumbers() }
        $reportDoc.Save()
        $reportDoc.ExportAsFixedFormat($pdfPath, 17)
        Write-Output ('Pages: ' + $reportDoc.ComputeStatistics(2))
        Write-Output $pdfPath
    }
    finally { $reportDoc.Close(0) }
}
finally { $wordApp.Quit() }
