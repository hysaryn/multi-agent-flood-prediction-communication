from bs4 import BeautifulSoup
from pathlib import Path
from .models import DocMeta
import fitz

def extract_text(meta: DocMeta) -> DocMeta:
    raw = Path(meta.raw_path)
    text_out = Path(meta.clean_path)

    if "pdf" in meta.content_type.lower() or raw.suffix.lower() == ".pdf":
        doc = fitz.open(raw)
        text = "\n".join(page.get_text() for page in doc)
    else:
        html = raw.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")
        title = soup.title.string.strip() if soup.title else None
        if title:
            meta.title = title
        text = soup.get_text(separator="\n")

    text_out.write_text(text.strip(), encoding="utf-8")
    return meta
