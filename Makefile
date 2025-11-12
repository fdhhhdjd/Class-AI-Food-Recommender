PYTHON = python3
VENV = .venv
ACTIVATE = source $(VENV)/bin/activate

install: $(VENV)/bin/activate
	@echo "📦 Installing dependencies..."
	@$(ACTIVATE) && pip install --upgrade pip && pip install -r requirements.txt
	@echo "✅ Installed"

$(VENV)/bin/activate:
	@echo "🧩 Creating virtual environment..."
	@$(PYTHON) -m venv $(VENV)

run-server: install
	@echo "🚀 Starting FastAPI server..."
	@.venv/bin/uvicorn server:app --reload --host 0.0.0.0 --port ${PORT:-8000}

precompute: install
	@echo "🔁 Precompute embeddings via API endpoint"
	@.venv/bin/python - <<'PY'
import requests
resp = requests.post('http://127.0.0.1:8000/api/precompute')
print(resp.text)
PY

clean:
	rm -rf $(VENV) __pycache__
	@echo "🧹 Cleaned"
