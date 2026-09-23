$ErrorActionPreference = 'Stop'

$docxPath = 'D:\develop\vscode\心理诊疗系统\artifacts\reports\五层心理评估系统_开发进程与Schema说明.docx'
$renderDir = 'D:\develop\vscode\心理诊疗系统\.tmp\l1_l5_report\final-render-v2'
$pdfPath = Join-Path $renderDir 'final.pdf'
$pdftoppm = 'C:\Users\ASUSA\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\poppler\Library\bin\pdftoppm.exe'

New-Item -ItemType Directory -Force -Path $renderDir | Out-Null
$word = New-Object -ComObject Word.Application
$word.Visible = $false
try {
    $doc = $word.Documents.Open($docxPath, $false, $true)
    $doc.ExportAsFixedFormat($pdfPath, 17)
    $doc.Close($false)
}
finally {
    $word.Quit()
}

& $pdftoppm -png -r 150 $pdfPath (Join-Path $renderDir 'page')
Get-ChildItem -LiteralPath $renderDir -Filter 'page-*.png' | Sort-Object Name | Select-Object -ExpandProperty FullName
