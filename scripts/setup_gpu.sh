#!/usr/bin/env bash
# Sensei hackathon — GPU machine setup (Ubuntu, RTX 4060 8GB VRAM)
# Run from the repo root. TIMEBOX: ~90 min total.
# If a step is stuck >20 min, screenshot it, move on, and flag it to the team.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo"

# Colours
CYAN='\033[0;36m'
YELLOW='\033[0;33m'
GREEN='\033[0;32m'
NC='\033[0m' # No Colour

step() { echo -e "\n${CYAN}=== $1 ===${NC}"; }
warn() { echo -e "${YELLOW}$1${NC}"; }
ok()   { echo -e "${GREEN}$1${NC}"; }

# ---------------------------------------------------------------------------
step "[1/6] NVIDIA driver"
# ---------------------------------------------------------------------------
if ! command -v nvidia-smi &>/dev/null; then
    warn "nvidia-smi not found. Install the NVIDIA driver first:"
    warn "  sudo ubuntu-drivers install          # auto-detect best driver"
    warn "  # or: sudo apt install nvidia-driver-560"
    warn "Then reboot and re-run this script."
    exit 1
fi
nvidia-smi 2>/dev/null | head -14 || true

# ---------------------------------------------------------------------------
step "[2/6] Ollama"
# ---------------------------------------------------------------------------
if ! command -v ollama &>/dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    # The installer puts ollama in /usr/local/bin or /usr/bin
    hash -r  # refresh PATH cache
fi
if ! command -v ollama &>/dev/null; then
    warn "ollama still not on PATH. Open a NEW terminal and re-run this script."
    exit 1
fi
ollama --version

# VRAM-optimized server settings for an 8GB GPU.
# A systemd service does NOT inherit shell exports, so when Ollama runs as a
# service we must write a drop-in override and restart it. Otherwise fall
# back to a local `ollama serve`, which does inherit the exports.
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_KEEP_ALIVE=-1
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q8_0

if systemctl list-unit-files 2>/dev/null | grep -q '^ollama\.service'; then
    echo "Ollama is a systemd service: applying drop-in override + restart..."
    sudo mkdir -p /etc/systemd/system/ollama.service.d
    sudo tee /etc/systemd/system/ollama.service.d/override.conf >/dev/null <<'EOF'
[Service]
Environment="OLLAMA_NUM_PARALLEL=1"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_KEEP_ALIVE=-1"
Environment="OLLAMA_FLASH_ATTENTION=1"
Environment="OLLAMA_KV_CACHE_TYPE=q8_0"
EOF
    sudo systemctl daemon-reload
    sudo systemctl restart ollama
else
    if ! curl -s http://localhost:11434/api/tags &>/dev/null; then
        echo "Starting Ollama server with VRAM-optimized settings..."
        nohup ollama serve > /tmp/ollama.log 2>&1 &
        sleep 2
    fi
fi
echo "Ollama reachable: $(curl -s http://localhost:11434/api/tags 2>/dev/null | head -c 80 || true)..."
echo "Service env: $(systemctl show ollama -p Environment 2>/dev/null || echo 'n/a (not a service)')"
echo "Check split with: ollama ps   (want 100% GPU)"

# ---------------------------------------------------------------------------
step "[3/6] Pull models (primary, fallback, second-opinion)"
# ---------------------------------------------------------------------------
echo "Pulling qwen3-vl:8b (primary, ~5GB)..."
ollama pull qwen3-vl:8b

echo "Pulling qwen3-vl:4b (fallback, more VRAM headroom)..."
ollama pull qwen3-vl:4b

echo "Pulling minicpm-v:8b (decorrelated second voter)..."
ollama pull minicpm-v:8b

ollama list

# ---------------------------------------------------------------------------
step "[4/6] Python GPU deps (venv)"
# ---------------------------------------------------------------------------
if [ ! -f ".venv/bin/python" ]; then
    echo "Creating virtualenv..."
    python3 -m venv .venv
fi

PY=".venv/bin/python"
PIP=".venv/bin/pip"

"$PIP" install --quiet --upgrade pip
"$PIP" install --quiet -r requirements.txt
"$PIP" install --quiet -r requirements-gpu.txt

"$PY" -c "import torch; print(f'torch {torch.__version__} | CUDA: {torch.cuda.is_available()} | {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU only\"}')"

# ---------------------------------------------------------------------------
step "[5/6] Pre-download DINOv2 (embedding features)"
# ---------------------------------------------------------------------------
"$PY" -c "
from transformers import AutoModel, AutoProcessor
AutoProcessor.from_pretrained('facebook/dinov2-small')
AutoModel.from_pretrained('facebook/dinov2-small')
print('dinov2-small cached OK')
"

# ---------------------------------------------------------------------------
step "[6/6] Smoke test: one VLM call"
# ---------------------------------------------------------------------------
ollama run qwen3-vl:8b "Reply with exactly one word: OK"

# ---------------------------------------------------------------------------
ok "=== DONE ==="
echo ""
echo "Next: drop the dataset into data/, then run the pipeline:"
echo "  .venv/bin/python scripts/run_pipeline.py --camera <id> --image <ts>.jpg"
echo "  make dry-run       # geometry + skeleton, no inference"
echo "  make test          # unit tests"
