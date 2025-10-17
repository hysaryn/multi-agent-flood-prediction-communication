from pathlib import Path
import hashlib, json, re, time
from datetime import datetime
from typing import Iterable, Dict
from .models import DocMeta

BASE = Path("action-plan-docs")
MANIFEST = BASE / "manifest.jsonl"

def slugify_place(place: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", place.strip().lower())
    return s.strip("-") or "unknown"

def place_dirs(place_key: str):
    place_slug = slugify_place(place_key)
    raw = BASE / place_slug / "raw"
    clean = BASE / place_slug / "clean"
    raw.mkdir(parents=True, exist_ok=True)
    clean.mkdir(parents=True, exist_ok=True)
    return raw, clean

def make_id(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]

def load_manifest() -> list[Dict]:
    if not MANIFEST.exists():
        return []
    with MANIFEST.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def write_manifest_line(meta: DocMeta) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST.open("a", encoding="utf-8") as f:
        f.write(meta.model_dump_json() + "\n")

def url_exists(url: str) -> Dict | None:
    # 简单线性查找；如数据多可换成 sqlite/键值索引
    for row in load_manifest():
        if row.get("url") == url:
            return row
    return None

def register_access(url: str) -> None:
    rows = load_manifest()
    changed = False
    for r in rows:
        if r.get("url") == url:
            r["last_accessed_at"] = datetime.utcnow().isoformat()
            changed = True
            break
    if changed:
        with MANIFEST.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

def build_paths(place_key: str, url: str, content_type: str) -> tuple[str, str]:
    raw_dir, clean_dir = place_dirs(place_key)
    stem = make_id(url)
    ext = ".pdf" if "pdf" in content_type.lower() or url.lower().endswith(".pdf") else ".html"
    return str(raw_dir / f"{stem}{ext}"), str(clean_dir / f"{stem}.txt")

# —— 清理策略 ——
def purge_place(place_key: str) -> int:
    """删除某个地点的 raw/clean 文件夹并从 manifest 中移除对应记录。"""
    target = BASE / place_key
    count = 0
    if target.exists():
        for p in target.rglob("*"):
            try:
                if p.is_file():
                    p.unlink()
                    count += 1
            except Exception:
                pass
        # 尝试删除空目录
        for p in sorted(target.rglob("*"), reverse=True):
            if p.is_dir():
                try: p.rmdir()
                except Exception: pass
        try: target.rmdir()
        except Exception: pass

    # 过滤 manifest
    rows = load_manifest()
    kept = [r for r in rows if r.get("place_key") != place_key]
    if len(kept) != len(rows):
        with MANIFEST.open("w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r) + "\n")
    return count

def list_places() -> list[str]:
    if not BASE.exists(): return []
    return [p.name for p in BASE.iterdir() if p.is_dir()]

def purge_all_except(keep_places: Iterable[str]) -> None:
    keep = set(keep_places)
    for place in list_places():
        if place not in keep:
            purge_place(place)

def purge_keep_last_n(n: int = 2) -> None:
    """按 last_accessed_at（地点粒度）保留最近 n 个地点，其余删除。"""
    rows = load_manifest()
    # 计算每个 place 的最新访问时间
    latest: dict[str, float] = {}
    for r in rows:
        pk = r.get("place_key")
        ts = r.get("last_accessed_at") or r.get("retrieved_at")
        try:
            t = datetime.fromisoformat(ts.replace("Z","")).timestamp()
        except Exception:
            t = 0.0
        latest[pk] = max(latest.get(pk, 0.0), t)
    ranked = sorted(latest.items(), key=lambda x: -x[1])
    keep = [pk for pk, _ in ranked[:max(0,n)]]
    purge_all_except(keep)
