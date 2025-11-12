# backend/server.py
from fastapi import FastAPI
from app.routes.recommend_route import router as recommend_router

app = FastAPI(title="AI-Recommend (modular)")

app.include_router(recommend_router)

# root quick check
@app.get("/")
def root():
    return {"ok": True, "msg": "AI-Recommend backend running"}
