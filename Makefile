PYTHON = python3
VENV = .venv
ACTIVATE = source $(VENV)/bin/activate
PORT ?= 8000  # đặt giá trị mặc định nếu không có PORT

install: $(VENV)/bin/activate
	@echo "📦 Installing dependencies..."
	@$(ACTIVATE) && pip install --upgrade pip && pip install -r requirements.txt
	@echo "✅ Installed"

$(VENV)/bin/activate:
	@echo "🧩 Creating virtual environment..."
	@$(PYTHON) -m venv $(VENV)

run-server: install
	@echo "🚀 Starting FastAPI server on port $(PORT)..."
	@.venv/bin/uvicorn server:app --reload --host 0.0.0.0 --port $(PORT)

precompute: install
	@echo "🔁 Precomputing embeddings via server endpoint..."
	@.venv/bin/python -c "import requests; \
r = requests.post('http://127.0.0.1:$(PORT)/api/precompute'); \
print(r.text)"

clean:
	@rm -rf $(VENV) __pycache__
	@echo "🧹 Cleaned"
