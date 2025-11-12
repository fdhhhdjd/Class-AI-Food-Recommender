# backend/app/controllers/recommend_controller.py
import numpy as np
from typing import List
from app.utils.io import load_items
from app.services.embed_service import build_index, precompute_and_cache
from app.models.item import RecommendItem

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)

def recommend(history_ids: List[int], top: int = 3, use_cache: bool = True):
    """
    Trả về list[RecommendItem] sắp xếp theo score giảm dần.
    """
    items = load_items()
    indexed = build_index(items, use_cache=use_cache)
    history_vecs = [it["vec"] for it in indexed if it["id"] in history_ids]
    if not history_vecs:
        raise ValueError("History empty or invalid ids")
    profile = np.stack(history_vecs, axis=0).mean(axis=0)
    cands = [it for it in indexed if it["id"] not in history_ids]
    scored = []
    for c in cands:
        s = cosine(profile, c["vec"])
        scored.append(RecommendItem(
            id=c["id"],
            name=c["name"],
            score=s,
            category=c.get("category"),
            price=c.get("price")
        ))
    scored.sort(key=lambda x: -x.score)
    return scored[:top]

def precompute():
    """
    Precompute embeddings for all items and save cache.
    Trả về số lượng item đã xử lý.
    """
    items = load_items()
    out = precompute_and_cache(items)
    return len(out)
