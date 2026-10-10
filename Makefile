.PHONY: setup run validate demo lint clean

setup:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	cp -n .env.example .env || true
	mkdir -p data/train data/test data/validation output cache demo

run:
	python3 scripts/run_pipeline.py --data-dir data/validation --output output/predictions.json

validate:
	python3 scripts/validate.py --predictions output/predictions.json --annotations data/annotations.json

demo:
	streamlit run app.py --server.port 8501

lint:
	ruff check src/ scripts/ app.py --fix
	ruff format src/ scripts/ app.py

clean:
	rm -rf cache/ output/ __pycache__ .ruff_cache
