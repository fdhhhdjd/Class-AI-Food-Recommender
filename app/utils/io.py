# backend/app/utils/io.py
import json
import numpy as np
from pathlib import Path
from app.config import ITEMS_FILE, CACHE_FILE

def load_items():
    """Đọc file gốc items.json"""
    if not ITEMS_FILE.exists():
        raise FileNotFoundError(f"Missing {ITEMS_FILE}")
    with open(ITEMS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def load_cached_items():
    """Đọc file cache (items_with_vecs.json). Trả về None nếu file không tồn tại hoặc invalid."""
    if not CACHE_FILE.exists():
        return None
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"WARNING: cache file {CACHE_FILE} invalid: {e}. Ignoring cache.")
        return None

    # convert vec lists to numpy arrays
    for it in raw:
        if "vec" in it and isinstance(it["vec"], list):
            it["vec"] = np.array(it["vec"], dtype=np.float32)
    return raw

def save_cached_items(items):
    """Lưu items + vec (chuyển numpy -> list) vào CACHE_FILE."""
    out = []
    for it in items:
        c = dict(it)
        if "vec" in c and hasattr(c["vec"], "tolist"):
            c["vec"] = c["vec"].tolist()
        out.append(c)
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"💾 Cache saved to {CACHE_FILE}")
