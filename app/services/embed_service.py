# backend/app/services/embed_service.py
import time
import numpy as np
from typing import List, Dict, Any
from app.config import client, MODEL
from app.utils.io import load_cached_items, save_cached_items

def embed_text(text: str, retries: int = 3, backoff: float = 2.0) -> np.ndarray:
    """
    Gọi Hugging Face InferenceClient.feature_extraction để lấy embedding.
    Trả về 1D numpy array. Retry nhẹ khi lỗi.
    """
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
    raise RuntimeError(f"Embedding failed after {retries} retries: {last_err}")


def precompute_and_cache(items: List[Dict[str, Any]], sleep_between: float = 0.12) -> List[Dict[str, Any]]:
    """
    Tính embedding cho tất cả items (dùng augmented desc nếu có) và lưu cache.
    Trả về list items (mỗi item có thêm 'vec' và 'aug_desc').
    """
    # build id->name for resolving pairs
    id2name = {it["id"]: it["name"] for it in items}

    out = []
    for d in items:
        it = dict(d)
        pairs = it.get("pair") or []
        pair_names = [id2name.get(pid) for pid in pairs if id2name.get(pid)]
        if pair_names:
            aug = f"{it.get('desc','')}. Món thường đi kèm: {', '.join(pair_names)}."
        else:
            aug = it.get('desc', '')
        it["aug_desc"] = aug
        print(f"→ Embedding id={it['id']} name={it['name']}")
        it["vec"] = embed_text(aug)
        out.append(it)
        time.sleep(sleep_between)
    save_cached_items(out)
    return out


def build_index(items: List[Dict[str, Any]], use_cache: bool = True) -> List[Dict[str, Any]]:
    """
    Trả về list items có 'vec' là numpy array và 'aug_desc'.
    Nếu use_cache True và cache hợp lệ sẽ dùng cache; nếu không, precompute và lưu cache.
    """
    if use_cache:
        cache = load_cached_items()
        if cache:
            print("✅ Using existing cache")
            # ensure aug_desc exists for compatibility (may be absent in older cache)
            for it in cache:
                if "aug_desc" not in it:
                    it["aug_desc"] = it.get("desc", "")
            return cache

    # fallback -> precompute & save
    return precompute_and_cache(items)
