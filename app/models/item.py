from typing import List, Optional
from pydantic import BaseModel

class Item(BaseModel):
    id: int
    name: str
    desc: str
    category: Optional[str] = None
    tags: Optional[List[str]] = []
    price: Optional[int] = None
    vec: Optional[List[float]] = None  # optional vector (for cache file)

class RecommendRequest(BaseModel):
    history: List[int]
    top: int = 3
    use_cache: bool = True

class RecommendItem(BaseModel):
    id: int
    name: str
    score: float
    category: Optional[str] = None
    price: Optional[int] = None
