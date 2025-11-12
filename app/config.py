from pathlib import Path
from dotenv import load_dotenv
import os
from huggingface_hub import InferenceClient

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
ITEMS_FILE = DATA_DIR / "items.json"
CACHE_FILE = DATA_DIR / "items_with_vecs.json"

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("Missing HF_TOKEN in .env")

MODEL = os.getenv("HF_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
HF_PROVIDER = os.getenv("HF_PROVIDER", "hf-inference")

# huggingface client
client = InferenceClient(provider=HF_PROVIDER, token=HF_TOKEN, timeout=60)
