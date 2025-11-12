from fastapi import APIRouter, HTTPException
from typing import List
from app.models.item import RecommendRequest
from app.controllers.recommend_controller import recommend, precompute
from app.services.embed_service import precompute_and_cache
from app.utils.io import load_items

router = APIRouter(prefix="/api", tags=["recommend"])

@router.get("/items")
def api_items():
    return load_items()

@router.post("/precompute")
def api_precompute():
    items = load_items()
    out = precompute_and_cache(items)
    return {"ok": True, "count": len(out)}

@router.post("/recommend")
def api_recommend(req: RecommendRequest):
    try:
        res = recommend(history_ids=req.history, top=req.top, use_cache=req.use_cache)
        return {"results": [r.dict() for r in res]}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
