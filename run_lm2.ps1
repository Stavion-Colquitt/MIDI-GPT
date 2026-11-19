# LM 2 (MIDI LM) runner: pins one GPU, runs/resumes/extends train.py and writes train_done.txt when finished.
#   New run:            .\run_lm2.ps1 -Hours 12
#   Resume:             .\run_lm2.ps1 -Resume
#   Continue training:  .\run_lm2.ps1 -ExtendHours 12 -Seed 2   (re-warm + new cosine from latest.pt)
param([double]$Hours = 12, [switch]$Resume, [double]$ExtendHours = 0, [int]$Seed = 1)
$here = Split-Path -Parent $MyInvocation.MyCommand.Path; Set-Location $here
$env:CUDA_DEVICE_ORDER = 'PCI_BUS_ID'      # (Nov 18) 0 = RTX 4060, 1 = RTX 5070
$env:CUDA_VISIBLE_DEVICES = '1'
$data = if ($env:LAKH_TOKENS) { $env:LAKH_TOKENS } else { Join-Path $here 'data' }   # token files from prep_midi.py
Remove-Item train_done.txt -ErrorAction SilentlyContinue
"$(Get-Date) run_lm2 Hours=$Hours Resume=$Resume ExtendHours=$ExtendHours Seed=$Seed" | Out-File run_lm2.log -Append
if ($ExtendHours -gt 0) { py -3.12 train.py --data $data --resume --extend_hours $ExtendHours --seed $Seed 1>> train.log 2>> train.err }
elseif ($Resume) { py -3.12 train.py --data $data --resume --seed $Seed 1>> train.log 2>> train.err }
else { py -3.12 train.py --data $data --hours $Hours 1> train.log 2> train.err }
if ($LASTEXITCODE -eq 0 -and ((Get-Content train.log -Tail 3) -match '^done\.')) { "ALL DONE $(Get-Date)" | Out-File train_done.txt }
