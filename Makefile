# Sensei pipeline — standard workflows
# Requires: make (Git for Windows includes it), Python venv
#
# Usage:
#   make test          # run unit tests
#   make dry-run       # geometry + skeleton, no inference
#   make run           # full pipeline (needs Ollama or GEMINI_API_KEY)
#   make explore       # dataset summary & sanity checks
#   make sheets        # contact sheets for all cameras (1st image each)
#   make demo          # launch Streamlit demo (needs: pip install streamlit)
#   make test          # run tests
#   make clean         # remove caches and outputs
#
# Override Python:  make PYTHON=/usr/bin/python3 test

# Auto-detect Python from venv (Windows vs Unix)
UNIX := $(shell uname 2>nul)
ifneq ($(UNIX),)
  PYTHON ?= .venv/bin/python
else
  PYTHON ?= .venv/Scripts/python.exe
endif

CAMERAS := 6215 6371 6373 6376 6383 6385 6386 6443 6462 6468 6491 6591 6594 6781

.PHONY: setup test dry-run run explore sheets demo clean lint

## Install dependencies into the venv
setup:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -r requirements-gpu.txt
	@echo "Setup complete. Run: make test"

## Full GPU machine setup (Ubuntu): driver check, Ollama, models, torch, DINOv2, smoke test
## Usage: make setup-gpu   (or: bash scripts/setup_gpu.sh)
setup-gpu:
	bash scripts/setup_gpu.sh

## Full GPU machine setup (Windows/PowerShell): driver check, Ollama, models, torch, DINOv2, smoke test
## Usage: make setup-gpu-win
setup-gpu-win:
	powershell -ExecutionPolicy Bypass -File scripts/setup_gpu.ps1

## Run unit tests
test:
	$(PYTHON) -m pytest tests/ -v

## Dry-run: geometry + skeleton + fallbacks (no API calls)
dry-run:
	$(PYTHON) scripts/run_pipeline.py --dry-run

## Full pipeline run (needs Ollama running or GEMINI_API_KEY set)
run:
	$(PYTHON) scripts/run_pipeline.py

## Dataset summary and sanity checks
explore:
	$(PYTHON) scripts/explore_data.py

## Generate contact sheets (first image per camera) for manual audit
sheets:
	@for cam in $(CAMERAS); do \
		$(PYTHON) scripts/contact_sheet.py --camera $$cam; \
	done
	@echo "Contact sheets written to sheets/"

## A/B compare backends against audit labels
## Usage: make ab LABELS=audit/labels.json PRED1=local=outputs/pred_local.json PRED2=api=outputs/pred_api.json
ab:
	$(PYTHON) scripts/ab_compare.py --labels $(LABELS) $(foreach p,$(PRED1) $(PRED2),--pred $(p))

## Launch Streamlit demo (tiebreaker UI)
demo:
	$(PYTHON) -m streamlit run app.py

## Lint / type-check
lint:
	$(PYTHON) -m py_compile sensei/*.py scripts/*.py
	@echo "Lint OK"

## Remove caches, outputs, and sheets
clean:
	rm -rf .cache/ outputs/ sheets/ __pycache__/** .pytest_cache/
	@echo "Cleaned."
