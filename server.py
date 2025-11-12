# server.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.recommend_route import router as recommend_router

app = FastAPI(title="AI Recommend API")

# CORS cho phép gọi từ Postman, browser, v.v.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả origin (an toàn trong dev)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gắn route chính (tất cả endpoints trong app/routes/recommend_route.py)
app.include_router(recommend_router, prefix="/api")

# Endpoint test nhanh
@app.get("/")
def root():
    return {"status": "ok"}
