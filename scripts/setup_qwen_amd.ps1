param(
  [string]$Model = "qwen3-vl:8b-instruct",
  [switch]$Vulkan
)
$ErrorActionPreference = "Stop"

# 1. Install Ollama if missing
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) {
  winget install -e --id Ollama.Ollama --accept-package-agreements --accept-source-agreements
  $env:Path = [Environment]::GetEnvironmentVariable("Path","Machine") + ";" +
              [Environment]::GetEnvironmentVariable("Path","User")
}
ollama --version

# 2. Environment (persisted for the user + applied to this session)
$vars = @{
  OLLAMA_NUM_PARALLEL      = "1"
  OLLAMA_MAX_LOADED_MODELS = "1"
  OLLAMA_KEEP_ALIVE        = "-1"
  OLLAMA_FLASH_ATTENTION   = "1"
  OLLAMA_KV_CACHE_TYPE     = "q8_0"
  HIP_VISIBLE_DEVICES      = "0"   # pick the dGPU if the CPU has an iGPU
}
if ($Vulkan) { $vars["OLLAMA_VULKAN"] = "1" }
foreach ($k in $vars.Keys) {
  [Environment]::SetEnvironmentVariable($k, $vars[$k], "User")
  Set-Item -Path "Env:$k" -Value $vars[$k]
}
if (-not $Vulkan) {
  [Environment]::SetEnvironmentVariable("OLLAMA_VULKAN", $null, "User")
  Remove-Item Env:OLLAMA_VULKAN -ErrorAction SilentlyContinue
}

# 3. Restart the server so it picks up the variables
Get-Process -Name "ollama*" -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 2
Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
for ($i = 0; $i -lt 30; $i++) {
  try { Invoke-RestMethod "http://localhost:11434/api/version" | Out-Null; break }
  catch { Start-Sleep -Seconds 1 }
}

# 4. Pull the model (~6 GB)
ollama pull $Model

# 5. Warm up with the SAME options your pipeline uses (otherwise Ollama reloads)
$body = @{
  model      = $Model
  prompt     = "Reply with OK."
  stream     = $false
  keep_alive = -1
  options    = @{ num_ctx = 4096; num_predict = 5 }
} | ConvertTo-Json -Depth 5
$sw = [Diagnostics.Stopwatch]::StartNew()
Invoke-RestMethod -Uri "http://localhost:11434/api/generate" -Method Post `
  -Body $body -ContentType "application/json" | Out-Null
Write-Host ("Warm-up took {0:N1}s" -f $sw.Elapsed.TotalSeconds)

# 6. Verify GPU usage
Write-Host "`n--- ollama ps ---"
$ps = ollama ps
$ps
Write-Host "`n--- backend lines from server.log ---"
$log = Join-Path $env:LOCALAPPDATA "Ollama\server.log"
if (Test-Path $log) {
  Select-String -Path $log -Pattern "gfx|rocm|hip|vulkan|offload|no compatible|inference compute" |
    Select-Object -Last 12 | ForEach-Object { $_.Line }
}

if ($ps -match "100% GPU") {
  Write-Host "`nOK: model is fully on the GPU." -ForegroundColor Green
} else {
  Write-Host "`nNot 100% GPU. Try: .\setup_qwen_amd.ps1 -Vulkan" -ForegroundColor Yellow
  Write-Host "Or fall back to the smaller model: .\setup_qwen_amd.ps1 -Model qwen3-vl:4b" -ForegroundColor Yellow
}