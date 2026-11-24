# LM 3 runner (RTX 4060 = GPU 0 ONLY). Waits for prep_lm3_done.txt on a fresh start, then fine-tunes from LM 2.
#   Fresh start:  .\run_lm3.ps1 -Hours 12
#   Resume:       .\run_lm3.ps1 -Resume          (after pause_lm3.flag, a crash or a reboot)
#   Pause:        create pause_lm3.flag in this folder (saves at the end of the current step, then exits)
param([double]$Hours = 12, [switch]$Resume)
$here = Split-Path -Parent $MyInvocation.MyCommand.Path; Set-Location $here
$env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'; $env:CUDA_VISIBLE_DEVICES = '0'; $env:PYTHONIOENCODING = 'utf-8'
$env:LM3_TOKENS = 'D:\Datasets\lakh\tokens_lm3'
"$(Get-Date) run_lm3 Hours=$Hours Resume=$Resume" | Out-File run_lm3.log -Append
if (-not $Resume) {
    while (-not (Test-Path prep_lm3_done.txt)) {
        if (-not (Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='py.exe'" | Where-Object CommandLine -match 'prep_lm3\.py')) {
            Start-Sleep 30
            if (-not (Test-Path prep_lm3_done.txt)) { "$(Get-Date) prep_lm3 is not running and not done; stopping" | Out-File run_lm3.log -Append; exit 1 }
        }
        Start-Sleep 60
    }
}
if (Test-Path pause_lm3.flag) { Rename-Item pause_lm3.flag ("pause_lm3.cleared_{0:MMdd_HHmm}.txt" -f (Get-Date)) }
Remove-Item lm3_done.txt -ErrorAction SilentlyContinue
if ($Resume) { py -3.12 train_lm3.py --resume 1>> train_lm3.log 2>> train_lm3.err }
else { py -3.12 train_lm3.py --hours $Hours 1> train_lm3.log 2> train_lm3.err }
"$(Get-Date) train_lm3 exit=$LASTEXITCODE" | Out-File run_lm3.log -Append
if ($LASTEXITCODE -eq 0 -and ((Get-Content train_lm3.log -Tail 1) -match '^done\.')) { "ALL DONE $(Get-Date)" | Out-File lm3_done.txt }
