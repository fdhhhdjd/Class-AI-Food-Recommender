import os, time, numpy as np
import requests  # vẫn giữ nếu bạn có chỗ khác dùng
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("Thiếu HF_TOKEN trong .env")

MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ⚠️ Dùng client chính thức: tự động gọi Router + đúng payload
client = InferenceClient(
    provider="hf-inference",   # ép dùng serverless HF mới
    token=HF_TOKEN,
    timeout=60
)

def embed(text: str, retries: int = 3, backoff: float = 2.0) -> np.ndarray:
    """
    Lấy embedding qua InferenceClient.feature_extraction (ổn định, không 400/410).
    Trả về vector 1D; nếu nhận token-level thì mean-pool.
    """
    last_err = None
    for i in range(retries):
        try:
            arr = client.feature_extraction(text, model=MODEL)  # -> np.ndarray
            arr = np.array(arr, dtype=np.float32)
            if arr.ndim == 1:
                return arr
            if arr.ndim == 2:
                return arr.mean(axis=0)
            raise RuntimeError(f"Định dạng embedding lạ: shape={arr.shape}")
        except Exception as e:
            last_err = e
            # Một số lúc provider warm-up → retry nhẹ
            time.sleep(backoff * (i + 1))
    raise RuntimeError(f"Embed thất bại sau {retries} lần: {last_err}")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)

# ----- DỮ LIỆU DEMO -----
dishes = [
  {"id":1, "name":"Phở bò",              "desc":"Phở nước dùng xương bò, bánh phở, thịt bò tái chín, rau thơm."},
  {"id":2, "name":"Bún chả",             "desc":"Thịt heo nướng than, bún tươi, nước mắm chua ngọt, rau sống."},
  {"id":3, "name":"Cơm tấm sườn bì chả", "desc":"Cơm tấm, sườn nướng, bì, chả trứng, mỡ hành."},
  {"id":4, "name":"Bún bò Huế",          "desc":"Bún sợi to, nước dùng cay nhẹ, giò heo, chả cua, mùi sả ớt."},
  {"id":5, "name":"Gỏi cuốn",            "desc":"Cuốn tôm thịt với bún, rau thơm, chấm nước mắm hoặc tương đậu."},
  {"id":6, "name":"Mì Quảng",            "desc":"Mì bản to, nước dùng ít, thịt heo/tôm, đậu phộng, bánh tráng nướng."},
  {"id":7, "name":"Cơm gà Hội An",       "desc":"Cơm nghệ, thịt gà xé, rau răm, nước mắm gừng."},
  {"id":8, "name":"Hủ tiếu Nam Vang",    "desc":"Nước trong, tôm thịt, gan, hủ tiếu dai, vị ngọt thanh."},
]
user_history_ids = [1, 4, 8]  # ví dụ user đã từng chọn các món này

def build_index(items):
    out = []
    for d in items:
        vec = embed(d["desc"])
        out.append({**d, "vec": vec})
    return out

def average_vec(vectors):
    if not vectors:
        raise RuntimeError("History rỗng.")
    mat = np.stack(vectors, axis=0)
    return mat.mean(axis=0)

def recommend_top_n(all_items, history_ids, top_n=3):
    with_vec = build_index(all_items)
    history_vecs = [d["vec"] for d in with_vec if d["id"] in history_ids]
    profile = average_vec(history_vecs)
    cands = [d for d in with_vec if d["id"] not in history_ids]
    scored = [{"name": c["name"], "score": cosine(profile, c["vec"])} for c in cands]
    scored.sort(key=lambda x: -x["score"])
    return scored[:top_n]

if __name__ == "__main__":
    recs = recommend_top_n(dishes, user_history_ids, top_n=3)
    print("Gợi ý top 3:")
    for r in recs:
        print(f"- {r['name']}  (similarity: {r['score']:.3f})")
