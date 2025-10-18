import requests
from datetime import datetime
from pathlib import Path
from ..models.govdoc_models import DocMeta
from .storage import build_paths, write_manifest_line, url_exists, register_access

def download(url: str, source: str, place_key: str) -> DocMeta:
    existed = url_exists(url)
    if existed:
        register_access(url)
        return DocMeta(**existed)

    # Add SSL handling
    r = requests.get(
        url, 
        timeout=25, 
        headers={"User-Agent": "flood-agent/1.0"},
        verify=True  # Try with verification first
    )
    r.raise_for_status()
    content_type = r.headers.get("content-type", "application/octet-stream")
    raw_path, clean_path = build_paths(place_key, url, content_type)

    Path(raw_path).write_bytes(r.content)
    meta = DocMeta(
        url=url,
        source=source,
        place_key=place_key,
        title=None,
        content_type=content_type,
        sha256=str(Path(raw_path).stat().st_size),    
        bytes=len(r.content),
        status=r.status_code,
        retrieved_at=datetime.utcnow(),
        last_accessed_at=datetime.utcnow(),
        raw_path=raw_path,
        clean_path=clean_path,
    )
    write_manifest_line(meta)
    return meta
