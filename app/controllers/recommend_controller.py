import numpy as np
from typing import List, Dict, Any
from app.utils.io import load_items
from app.services.embed_service import build_index, precompute_and_cache
from app.models.item import RecommendItem

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)

def recommend(history_ids: List[int],
              top: int = 3,
              use_cache: bool = True,
              category_boost: float = 1.0,
              pair_boost: float = 0.15) -> List[RecommendItem]:
    """
    Trả về list[RecommendItem] đã sắp xếp.
    - use_cache: nếu True sẽ dùng data/items_with_vecs.json nếu có.
    - category_boost: multiplier cho candidate có cùng category với history.
    - pair_boost: additive boost nếu candidate được khai báo trong 'pair' của bất kỳ history item.
    """
    items = load_items()
    indexed = build_index(items, use_cache=use_cache)

    id_map: Dict[int, Dict[str, Any]] = {it["id"]: it for it in indexed}

    history_vecs = []
    history_categories = set()
    declared_pairs = set()
    for hid in history_ids:
        it = id_map.get(hid)
        if not it:
            continue
        history_vecs.append(it["vec"])
        if it.get("category"):
            history_categories.add(it["category"])
        for pid in it.get("pair", []) or []:
            declared_pairs.add(pid)

    if not history_vecs:
        raise ValueError("History empty or invalid ids")

    profile = np.stack(history_vecs, axis=0).mean(axis=0)

    scored = []
    for c in indexed:
        cid = c["id"]
        if cid in history_ids:
            continue
        base_score = cosine(profile, c["vec"])

        # category multiplier (multiply)
        if history_categories and c.get("category") in history_categories:
            base_score = base_score * category_boost

        # pair additive boost (cap < 1)
        if cid in declared_pairs:
            base_score = min(base_score + pair_boost, 0.9999)

        scored.append(RecommendItem(
            id=cid,
            name=c["name"],
            score=base_score,
            category=c.get("category"),
            price=c.get("price")
        ))

    scored.sort(key=lambda x: -x.score)
    return scored[:top]


def precompute() -> int:
    """
    Controller helper: precompute embeddings for all items and return count.
    """
    items = load_items()
    out = precompute_and_cache(items)
    return len(out)
