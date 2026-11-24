# LM 3 data prep (CPU only). Writes D:\Datasets\lakh\tokens_lm3 and prep_lm3_done.txt.
$here = Split-Path -Parent $MyInvocation.MyCommand.Path; Set-Location $here
$env:PYTHONIOENCODING = 'utf-8'
Remove-Item prep_lm3_done.txt -ErrorAction SilentlyContinue
"$(Get-Date) prep_lm3 start" | Out-File run_lm3.log -Append
py -3.12 prep_lm3.py --workers 6 1> prep_lm3.log 2> prep_lm3.err
if ($LASTEXITCODE -eq 0 -and ((Get-Content prep_lm3.log -Tail 1) -match 'PREP DONE')) { "PREP DONE $(Get-Date)" | Out-File prep_lm3_done.txt }
"$(Get-Date) prep_lm3 end exit=$LASTEXITCODE" | Out-File run_lm3.log -Append
