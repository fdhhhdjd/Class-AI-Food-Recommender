import time
import numpy as np
from app.config import client, MODEL
from typing import List, Dict, Any
from app.utils.io import load_cached_items, save_cached_items

def embed_text(text: str, retries: int = 3, backoff: float = 2.0) -> np.ndarray:
    last_err = None
    for i in range(retries):
        try:
            arr = client.feature_extraction(text, model=MODEL)
            arr = np.array(arr, dtype=np.float32)
            if arr.ndim == 2:
                return arr.mean(axis=0)
            return arr
        except Exception as e:
            last_err = e
            time.sleep(backoff * (i + 1))
    raise RuntimeError(f"Embed failed: {last_err}")

def precompute_and_cache(items: List[Dict[str,Any]], sleep_between: float = 0.25):
    out = []
    for it in items:
        vec = embed_text(it["desc"])
        copy = dict(it)
        copy["vec"] = vec  # numpy array; save_cached_items will convert
        out.append(copy)
        time.sleep(sleep_between)
    save_cached_items(out)
    return out

def build_index(items: List[Dict[str,Any]], use_cache: bool = True):
    """
    Return list of items where each has 'vec' as numpy array.
    If cache exists and use_cache True, will merge vectors from cache.
    """
    cache = load_cached_items() if use_cache else None
    cache_map = {it["id"]: it for it in cache} if cache else {}
    out = []
    for d in items:
        it = dict(d)
        if d.get("id") in cache_map and "vec" in cache_map[d["id"]]:
            it["vec"] = cache_map[d["id"]]["vec"]
        elif "vec" in it:
            it["vec"] = np.array(it["vec"], dtype=np.float32)
        else:
            it["vec"] = embed_text(it["desc"])
        out.append(it)
    return out
