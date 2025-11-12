from fastapi import APIRouter
from app.utils.io import load_items
from app.controllers.recommend_controller import recommend, precompute

# ⚠️ Không có prefix ở đây!
router = APIRouter(tags=["Recommend"])

@router.get("/items")
def api_items():
    return load_items()

@router.post("/recommend")
def api_recommend(data: dict):
    try:
        history = data.get("history", [])
        top = data.get("top", 3)
        use_cache = data.get("use_cache", True)
        category_boost = float(data.get("category_boost", 1.0))
        pair_boost = float(data.get("pair_boost", 0.15))
        res = recommend(history, top, use_cache, category_boost, pair_boost)
        return {"results": res}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/precompute")
def api_precompute():
    count = precompute()
    return {"ok": True, "count": count}
