from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_text_splitters import RecursiveCharacterTextSplitter

def normalize_section(title: str) -> str:
    t = (title or "").lower().replace("\u00a0", " ").strip()
    # keep just the "item x." prefix when possible
    for key in ["item 1a.", "item 1.", "item 2.", "item 3.", "item 4.",
                "item 7a.", "item 7.", "item 8."]:
        if t.startswith(key):
            return key
    return t[:40]  # short fallback

def split_sections(section_dicts, chunk_size=1200, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " "]
    )
    out = []
    for sec in section_dicts:
        sec_title = sec.get("title","").strip()
        sec_norm  = normalize_section(sec_title)
        texts = splitter.split_text(sec["text"])
        for ch in texts:
            out.append({
                "title": sec_title,
                "section": sec_norm,
                "text": ch
            })
    return out


