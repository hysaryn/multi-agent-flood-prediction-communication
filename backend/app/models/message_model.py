from pydantic import BaseModel
from typing import Any, Dict, Optional

class Message(BaseModel):
    content: str
    metadata: Optional[Dict[str, Any]] = None
