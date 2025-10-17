from pydantic import BaseModel, HttpUrl
from datetime import datetime

class DocMeta(BaseModel):
    url: HttpUrl
    source: str
    place_key: str 
    title: str | None = None
    content_type: str
    sha256: str            # 基于URL或内容的指纹
    bytes: int
    status: int
    retrieved_at: datetime
    last_accessed_at: datetime
    raw_path: str          # backend/action plan docs/raw/xxx
    clean_path: str        # backend/action plan docs/clean/xxx

class DocRef(BaseModel):
    url: HttpUrl
    title: str | None = None
    clean_path: str
    place_key: str
