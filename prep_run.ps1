# LM 2 data prep runner: wait for the Lakh download, extract, train the BPE tokenizer + time a test, then the full encode.
$h = $PSScriptRoot; Set-Location $h
function Log($m) { "$(Get-Date -Format 'HH:mm:ss') $m" | Out-File "$h\prep.log" -Append }
while (Get-Process curl -ErrorAction SilentlyContinue) { Start-Sleep 10 }
Log "download done: $((Get-Item data\lmd_full.tar.gz).Length) bytes"
if (-not (Test-Path data\lmd_full)) { tar -xzf data\lmd_full.tar.gz -C data 2>> prep.err; Log "extracted" }
py -3.12 prep_midi.py --test 400 --workers 6 1>> prep.log 2>> prep.err
Log "test done; starting full encode"
py -3.12 prep_midi.py --workers 6 1>> prep.log 2>> prep.err
Log "PREP DONE"
