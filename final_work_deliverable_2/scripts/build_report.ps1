# Build the Deliverable 2 report from the frozen tables.
# Run from anywhere:
#   powershell -ExecutionPolicy Bypass -File scripts/build_report.ps1
# If Tectonic or pdflatex is installed, the LaTeX source is compiled.
# Use -HtmlFallback only when no LaTeX engine is available.
param(
    [string]$Tectonic = $env:TECTONIC,
    [switch]$HtmlFallback
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root

Write-Host 'Building tables...'
python scripts/build_tables.py

Write-Host 'Building architecture diagram...'
python scripts/build_architecture.py

Write-Host 'Building HTML companion...'
python scripts/build_html.py

$tex = Join-Path $root 'report_d2.tex'
$pdf = Join-Path $root 'report_d2.pdf'

if (-not $HtmlFallback) {
    $tectonicExe = $null
    if ($Tectonic -and (Test-Path -LiteralPath $Tectonic)) {
        $tectonicExe = $Tectonic
    }
    if (-not $tectonicExe) {
        $command = Get-Command tectonic -ErrorAction SilentlyContinue
        if ($command) { $tectonicExe = $command.Source }
    }

    if ($tectonicExe) {
        Write-Host "Compiling $tex with Tectonic..."
        if (Test-Path -LiteralPath $pdf) { Remove-Item -LiteralPath $pdf -Force }
        & $tectonicExe -X compile $tex --keep-logs
        if ($LASTEXITCODE -ne 0) { throw 'Tectonic failed to build report_d2.pdf.' }
        if (Test-Path -LiteralPath 'report_d2.log') {
            Remove-Item -LiteralPath 'report_d2.log' -Force
        }
        Get-Item -LiteralPath $pdf | Select-Object FullName, Length, LastWriteTime
        Write-Host 'LaTeX report build complete.'
        exit 0
    }

    $pdflatex = Get-Command pdflatex -ErrorAction SilentlyContinue
    if ($pdflatex) {
        Write-Host "Compiling $tex with pdflatex..."
        & $pdflatex.Source -interaction=nonstopmode -halt-on-error report_d2.tex | Out-Null
        if ($LASTEXITCODE -ne 0) { throw 'pdflatex failed to build report_d2.pdf (first pass).' }
        & $pdflatex.Source -interaction=nonstopmode -halt-on-error report_d2.tex | Out-Null
        if ($LASTEXITCODE -ne 0) { throw 'pdflatex failed to build report_d2.pdf (second pass).' }
        Get-Item -LiteralPath $pdf | Select-Object FullName, Length, LastWriteTime
        Write-Host 'LaTeX report build complete.'
        exit 0
    }

    throw 'No LaTeX engine found. Install Tectonic or pdflatex, or rerun with -HtmlFallback to use the HTML print path.'
}

$html = Join-Path $root 'report_d2.html'
$fallbackPdf = Join-Path $root 'report_d2_html.pdf'

$edgeCandidates = @(
    'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
    'C:\Program Files\Microsoft\Edge\Application\msedge.exe'
)
$edge = $edgeCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1

if (-not $edge) {
    throw 'Microsoft Edge was not found. Open report_d2.html manually and print it to PDF.'
}

if (Test-Path -LiteralPath $fallbackPdf) {
    Remove-Item -LiteralPath $fallbackPdf -Force
}

$uri = [System.Uri]::new($html).AbsoluteUri
Write-Host "Printing $html to $fallbackPdf ..."
& $edge '--headless' '--disable-gpu' '--no-pdf-header-footer' '--allow-file-access-from-files' "--print-to-pdf=$fallbackPdf" $uri 2>$null
Start-Sleep -Seconds 2

if (-not (Test-Path -LiteralPath $fallbackPdf)) {
    throw "HTML fallback PDF was not created at $fallbackPdf"
}

if (-not (Test-Path -LiteralPath $pdf)) {
    Copy-Item -LiteralPath $fallbackPdf -Destination $pdf
    Write-Host 'No LaTeX PDF was present; copied the HTML fallback to report_d2.pdf.'
}

Get-Item -LiteralPath $fallbackPdf | Select-Object FullName, Length, LastWriteTime
Write-Host 'HTML fallback build complete.'
