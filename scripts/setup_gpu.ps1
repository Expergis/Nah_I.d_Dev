# Sensei hackathon — GPU machine setup (RTX 4060, 8GB VRAM)
# Run from the repo root. TIMEBOX: ~90 min total. If a step is stuck >20 min,
# screenshot it, move on, and flag it to the team.
$ErrorActionPreference = "Continue"
$repo = $PSScriptRoot | Split-Path -Parent
Set-Location $repo

Write-Host "`n=== [1/6] NVIDIA driver ===" -ForegroundColor Cyan
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    Write-Host "nvidia-smi not found — install the NVIDIA driver first (nvidia.com)." -ForegroundColor Yellow
    exit 1
}
nvidia-smi | Select-Object -First 14

Write-Host "`n=== [2/6] Ollama ===" -ForegroundColor Cyan
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    winget install --id Ollama.Ollama -e --accept-source-agreements --accept-package-agreements
    foreach ($p in @("$env:LOCALAPPDATA\Programs\Ollama", "$env:LOCALAPPDATA\Programs\ollama", "C:\Program Files\Ollama")) {
        if (Test-Path (Join-Path $p "ollama.exe")) { $env:Path += ";$p" }
    }
}
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
    Write-Host "ollama still not on PATH — open a NEW terminal and re-run this script." -ForegroundColor Yellow
    exit 1
}
ollama --version
# VRAM-optimized settings for 8GB GPU (set before ollama serve)
$env:OLLAMA_NUM_PARALLEL = "1"
$env:OLLAMA_MAX_LOADED_MODELS = "1"
$env:OLLAMA_KEEP_ALIVE = "-1"
$env:OLLAMA_FLASH_ATTENTION = "1"
$env:OLLAMA_KV_CACHE_TYPE = "q8_0"

ollama serve *> $null &  # no-op guard if already running; the service usually autostarts

Write-Host "`n=== [3/6] Pull models (primary, fallback, second-opinion) ===" -ForegroundColor Cyan
ollama pull qwen3-vl:8b   # primary (~5GB Q4)
ollama pull qwen3-vl:4b   # fallback, more VRAM headroom
ollama pull minicpm-v:8b  # decorrelated second voter (optional)
ollama list

Write-Host "`n=== [4/6] Python GPU deps (venv) ===" -ForegroundColor Cyan
if (-not (Test-Path ".venv\Scripts\python.exe")) { python -m venv .venv }
& ".venv\Scripts\python.exe" -m pip install --quiet --upgrade pip
& ".venv\Scripts\python.exe" -m pip install --quiet -r requirements-gpu.txt
& ".venv\Scripts\python.exe" -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"

Write-Host "`n=== [5/6] Pre-download DINOv2 (embedding features) ===" -ForegroundColor Cyan
& ".venv\Scripts\python.exe" -c "from transformers import AutoModel, AutoProcessor; AutoProcessor.from_pretrained('facebook/dinov2-small'); AutoModel.from_pretrained('facebook/dinov2-small'); print('dinov2-small cached OK')"

Write-Host "`n=== [6/6] Smoke test: one VLM call ===" -ForegroundColor Cyan
ollama run qwen3-vl:8b "Reply with exactly one word: OK"

Write-Host "`n=== DONE ===" -ForegroundColor Green
Write-Host "Next: drop the dataset into data/, then run the pipeline:"
Write-Host "  .venv\Scripts\python.exe scripts\run_pipeline.py --camera <id> --image <ts>.jpg"
