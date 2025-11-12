#!/usr/bin/env python3
import os
import time
import json
import argparse
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# ----- config -----
DATA_DIR = Path("data")
ITEMS_FILE = DATA_DIR / "items.json"               # your external data file
CACHE_FILE = DATA_DIR / "items_with_vecs.json"     # generated cache

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("Thiếu HF_TOKEN trong .env (xem .env.example)")

MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# official client -> robust with new router
client = InferenceClient(provider="hf-inference", token=HF_TOKEN, timeout=60)


# ----- embedding helper -----
def embed(text: str, retries: int = 3, backoff: float = 2.0) -> np.ndarray:
    last_err = None
    for i in range(retries):
        try:
            arr = client.feature_extraction(text, model=MODEL)
            arr = np.array(arr, dtype=np.float32)
            if arr.ndim == 1:
                return arr
            if arr.ndim == 2:
                return arr.mean(axis=0)
            raise RuntimeError(f"Định dạng embedding lạ: shape={arr.shape}")
        except Exception as e:
            last_err = e
            time.sleep(backoff * (i + 1))
    raise RuntimeError(f"Embed thất bại sau {retries} lần: {last_err}")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


# ----- I/O: load items / cache -----
def load_items():
    if not ITEMS_FILE.exists():
        raise FileNotFoundError(f"Không tìm thấy {ITEMS_FILE}. Put your items.json into data/ folder.")
    with open(ITEMS_FILE, "r", encoding="utf-8") as f:
        items = json.load(f)
    return items


def save_cached_items(items_with_vecs):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for it in items_with_vecs:
        copy = dict(it)
        if "vec" in copy and isinstance(copy["vec"], np.ndarray):
            copy["vec"] = copy["vec"].tolist()
        out.append(copy)
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved cache: {CACHE_FILE}")


def load_cached_items():
    if not CACHE_FILE.exists():
        return None
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        raw = json.load(f)
    for it in raw:
        if "vec" in it and isinstance(it["vec"], list):
            it["vec"] = np.array(it["vec"], dtype=np.float32)
    return raw


# ----- Build index (ensure vec present per item) -----
def build_index(items, use_cache=True):
    cached = load_cached_items() if use_cache else None
    cache_map = {it["id"]: it for it in cached} if cached else {}

    out = []
    for d in items:
        it = dict(d)
        if d.get("id") in cache_map and "vec" in cache_map[d["id"]]:
            it["vec"] = cache_map[d["id"]]["vec"]
        elif "vec" in d:
            it["vec"] = np.array(d["vec"], dtype=np.float32)
        else:
            print(f"→ Computing embed for id={d.get('id')} name={d.get('name')}")
            it["vec"] = embed(d["desc"])
        out.append(it)
    return out


# ----- recommend logic -----
def average_vec(vectors):
    if not vectors:
        raise RuntimeError("History rỗng.")
    mat = np.stack(vectors, axis=0)
    return mat.mean(axis=0)


def recommend_top_n(all_items, history_ids, top_n=3):
    with_vec = build_index(all_items, use_cache=True)
    history_vecs = [d["vec"] for d in with_vec if d["id"] in history_ids]
    if not history_vecs:
        raise RuntimeError("Không tìm thấy item nào trong history_ids.")
    profile = average_vec(history_vecs)
    cands = [d for d in with_vec if d["id"] not in history_ids]
    scored = [{"id": c["id"], "name": c["name"], "score": cosine(profile, c["vec"]), "price": c.get("price"), "category": c.get("category")} for c in cands]
    scored.sort(key=lambda x: -x["score"])
    return scored[:top_n]


# ----- precompute embeddings for all items and write cache -----
def precompute_embeddings(items, sleep_between=0.5):
    print("🔁 Precomputing embeddings for all items (this may take a while)...")
    out = []
    for d in items:
        print(f" - id={d.get('id')} name={d.get('name')}")
        vec = embed(d["desc"])
        it = dict(d)
        it["vec"] = vec
        out.append(it)
        time.sleep(sleep_between)  # gentle spacing to avoid throttling
    save_cached_items(out)
    return out


# ----- CLI -----
def parse_args():
    p = argparse.ArgumentParser(description="AI Recommend - using Hugging Face embeddings (cacheable)")
    p.add_argument("--precompute", action="store_true", help="Precompute embeddings for all items and save to data/items_with_vecs.json")
    p.add_argument("--no-cache", action="store_true", help="Do not use cache when running recommendation (compute on-the-fly)")
    p.add_argument("--history", nargs="+", type=int, default=[1, 4, 8], help="List of item ids representing user history")
    p.add_argument("--top", type=int, default=3, help="Top N recommendations to show")
    return p.parse_args()


def main():
    args = parse_args()
    items = load_items()

    if args.precompute:
        precompute_embeddings(items)
        return

    if not CACHE_FILE.exists():
        print("⚠️  Cache (data/items_with_vecs.json) not found. Recommend to run with --precompute once to save quota.")
    index_items = build_index(items, use_cache=not args.no_cache)
    recs = recommend_top_n(index_items, history_ids=args.history, top_n=args.top)
    print("\nGợi ý top", args.top)
    for r in recs:
        print(f"- {r['name']}  (score: {r['score']:.4f}) • {r.get('category','')} • {r.get('price','')}")
    print()


if __name__ == "__main__":
    main()
