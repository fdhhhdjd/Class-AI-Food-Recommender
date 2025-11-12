# backend/app/utils/io.py
import json
from pathlib import Path
import numpy as np
from app.config import ITEMS_FILE, CACHE_FILE

def load_items():
    if not ITEMS_FILE.exists():
        raise FileNotFoundError(f"Missing {ITEMS_FILE}")
    with open(ITEMS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def load_cached_items():
    if not CACHE_FILE.exists():
        return None
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)
    # convert vec lists to numpy arrays for internal use
    for it in raw:
        if "vec" in it and isinstance(it["vec"], list):
            it["vec"] = np.array(it["vec"], dtype="float32")
    return raw

def save_cached_items(items):
    out = []
    for it in items:
        c = dict(it)
        if "vec" in c and hasattr(c["vec"], "tolist"):
            c["vec"] = c["vec"].tolist()
        out.append(c)
    DATA_DIR = Path(CACHE_FILE).parent
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
