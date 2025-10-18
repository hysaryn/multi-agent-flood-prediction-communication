from pydantic import BaseModel, HttpUrl
from datetime import datetime

class DocMeta(BaseModel):
    url: HttpUrl
    source: str
    place_key: str 
    title: str | None = None
    content_type: str
    sha256: str             
    bytes: int
    status: int
    retrieved_at: datetime
    last_accessed_at: datetime
    raw_path: str           
    clean_path: str         

class DocRef(BaseModel):
    url: HttpUrl
    title: str | None = None
    clean_path: str
    place_key: str
