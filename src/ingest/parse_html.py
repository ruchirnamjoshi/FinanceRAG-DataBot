# src/ingest/parse_html.py
from bs4 import BeautifulSoup
from bs4 import XMLParsedAsHTMLWarning
import warnings, re

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

ITEM_PAT  = re.compile(r"\bitem\s+(1a|1|2|3|4|7a|7|8)\b\.?", re.I)
MDNA_PAT  = re.compile(r"management[’']?s?\s+discussion\s+and\s+analysis", re.I)
NBSP      = "\u00a0"

def _normalize_ws(s: str) -> str:
    return re.sub(r"[ \t"+NBSP+r"]+", " ", s or "").strip()

def _blocks_from_html(soup: BeautifulSoup):
    for t in soup(["script","style","noscript"]): t.decompose()
    blocks = []
    for tag in soup.find_all(["h1","h2","h3","h4","p","div","table","li"]):
        txt = _normalize_ws(tag.get_text(" ", strip=True))
        if txt:
            blocks.append(txt)
    # fallback: whole doc text
    if not blocks:
        txt = _normalize_ws(soup.get_text(" ", strip=True))
        if txt:
            blocks = [txt]
    return blocks

def html_to_sections(html_text: str):
    text_head = html_text.lstrip()[:200].lower()

    # 1) Choose parser: XML if it truly starts as XML/XBRL, else HTML
    if text_head.startswith("<?xml") or text_head.startswith("<xbrl"):
        soup = BeautifulSoup(html_text, "xml")
        full = _normalize_ws(soup.get_text(" ", strip=True))
        if not full:
            return []
        # minimal sectionization around MD&A if present
        secs = []
        mdna = list(MDNA_PAT.finditer(full))
        if mdna:
            last = 0
            for m in mdna:
                pre = full[last:m.start()].strip()
                if pre:
                    secs.append({"title":"Document (pre-MD&A)", "text":pre})
                secs.append({"title":"Item 2. Management’s Discussion and Analysis", "text": full[m.start():m.end()+3000]})
                last = m.end()+3000
            tail = full[last:].strip()
            if tail:
                secs.append({"title":"Document (post-MD&A)", "text": tail})
            return secs
        else:
            # single-section fallback
            return [{"title":"Document (XML)", "text": full}]

    # HTML path
    soup = BeautifulSoup(html_text, "lxml")
    blocks = _blocks_from_html(soup)

    # 2) Try splitting on explicit Item headers
    sections = []
    cur = {"title":"Document Start", "text":""}
    found_item = False

    def is_item_header(line: str) -> bool:
        return bool(ITEM_PAT.search(line.lower().replace(NBSP, " ")))

    for line in blocks:
        if is_item_header(line):
            found_item = True
            if cur["text"].strip():
                sections.append(cur)
            cur = {"title": _normalize_ws(line), "text": ""}
        else:
            cur["text"] += line + "\n"
    if cur["text"].strip():
        sections.append(cur)

    # 3) If no Items found, try to carve around MD&A anchor
    if not found_item:
        full = "\n".join(blocks)
        mdna = list(MDNA_PAT.finditer(full))
        if mdna:
            sections = []
            last = 0
            for m in mdna:
                pre = full[last:m.start()].strip()
                if pre:
                    sections.append({"title":"Document (pre-MD&A)", "text":pre})
                sections.append({"title":"Item 2. Management’s Discussion and Analysis", "text": full[m.start():m.end()+3000]})
                last = m.end()+3000
            tail = full[last:].strip()
            if tail:
                sections.append({"title":"Document (post-MD&A)", "text": tail})

    # 4) Absolute fallback: whole doc as one section
    if not sections:
        full = "\n".join(blocks)
        if full.strip():
            sections = [{"title":"Document", "text": full}]

    return sections
