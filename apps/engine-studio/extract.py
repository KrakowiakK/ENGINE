#!/usr/bin/env python3
"""Turns an uploaded file or a fetched web page into plain text the model can read as context.

No inference here, no champion involvement -- this is pure text extraction, injected into a chat
message as a fenced block (see server.py's ATTACHMENT_TEMPLATE). PDF support comes from `pypdf`,
kept in `.venv/` (a local virtualenv, not the system Python this app otherwise runs under) so the
control server stays dependency-free for everything else; DOCX/XLSX are parsed by hand from their
own XML with the standard library only, since both formats are just a zip of XML.
"""
import html.parser
import os
import re
import sys
import zipfile
from xml.etree import ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_CHARS = 100_000   # a generous cap: ~25-30K tokens, well inside the model's context but not free

TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".yaml", ".yml", ".xml", ".log",
    ".py", ".js", ".ts", ".jsx", ".tsx", ".swift", ".c", ".h", ".cpp", ".hpp", ".java",
    ".go", ".rs", ".sh", ".rb", ".php", ".sql", ".ini", ".cfg", ".toml", ".rst", ".html", ".htm",
}


def _truncate(text):
    text = text.strip()
    if len(text) > MAX_CHARS:
        return text[:MAX_CHARS] + f"\n\n[... truncated, {len(text) - MAX_CHARS} more characters not shown ...]", True
    return text, False


def _pdf_sitepackages():
    d = os.path.join(HERE, ".venv", "lib")
    if not os.path.isdir(d):
        return None
    for name in os.listdir(d):
        sp = os.path.join(d, name, "site-packages")
        if os.path.isdir(sp):
            return sp
    return None


def extract_pdf(data):
    sp = _pdf_sitepackages()
    if sp is None:
        raise RuntimeError("PDF support needs the local virtualenv: cd apps/engine-studio && "
                            "python3 -m venv .venv && .venv/bin/pip install pypdf")
    if sp not in sys.path:
        sys.path.insert(0, sp)
    import io
    import pypdf
    reader = pypdf.PdfReader(io.BytesIO(data))
    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception:
            raise RuntimeError("this PDF is password-protected")
    parts = []
    for i, page in enumerate(reader.pages):
        t = page.extract_text() or ""
        if t.strip():
            parts.append(f"--- page {i + 1} ---\n{t.strip()}")
    return "\n\n".join(parts)


def extract_docx(data):
    import io
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        xml = z.read("word/document.xml")
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    root = ET.fromstring(xml)
    paragraphs = []
    for p in root.iter(f"{{{ns['w']}}}p"):
        runs = [t.text or "" for t in p.iter(f"{{{ns['w']}}}t")]
        line = "".join(runs)
        if line.strip():
            paragraphs.append(line)
    return "\n".join(paragraphs)


def extract_xlsx(data):
    import io
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.iter(f"{{{ns['s']}}}si"):
                shared.append("".join(t.text or "" for t in si.iter(f"{{{ns['s']}}}t")))
        sheet_names = sorted(n for n in z.namelist() if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n))
        ns = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        out = []
        for sn in sheet_names:
            root = ET.fromstring(z.read(sn))
            rows = []
            for row in root.iter(f"{{{ns['s']}}}row"):
                cells = []
                for c in row.iter(f"{{{ns['s']}}}c"):
                    v = c.find(f"{{{ns['s']}}}v")
                    if v is None or v.text is None:
                        cells.append("")
                        continue
                    if c.get("t") == "s":
                        idx = int(v.text)
                        cells.append(shared[idx] if 0 <= idx < len(shared) else "")
                    else:
                        cells.append(v.text)
                rows.append("\t".join(cells))
            if rows:
                out.append(f"--- {sn.split('/')[-1]} ---\n" + "\n".join(rows))
        return "\n\n".join(out)


def extract_file(filename, data):
    """(text, truncated) or raises RuntimeError with a message safe to show the user."""
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        text = extract_pdf(data)
    elif ext == ".docx":
        text = extract_docx(data)
    elif ext == ".xlsx":
        text = extract_xlsx(data)
    elif ext in (".doc", ".xls", ".ppt", ".pptx"):
        raise RuntimeError(f"{ext} is not supported (old binary Office formats, or slides) -- "
                            "save as .docx / .xlsx / .txt and try again")
    elif ext in TEXT_EXTENSIONS or ext == "":
        text = data.decode("utf-8", errors="replace")
    else:
        # unknown extension: try UTF-8 text as a last resort rather than refusing outright
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            raise RuntimeError(f"don't know how to read {ext or '(no extension)'} files")
    if not text.strip():
        raise RuntimeError("no extractable text found in this file")
    return _truncate(text)


# --------------------------------------------------------------------------------- web pages

class _TextHTMLParser(html.parser.HTMLParser):
    """Visible text only: drops <script>/<style>/<head>, keeps a rough line structure."""
    SKIP_TAGS = {"script", "style", "noscript", "head", "svg", "template"}
    BLOCK_TAGS = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.chunks = []
        self.title = ""
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True
        if tag in self.BLOCK_TAGS:
            self.chunks.append("\n")

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data
            return
        if self._skip_depth:
            return
        self.chunks.append(data)

    def text(self):
        raw = "".join(self.chunks)
        lines = [ln.strip() for ln in raw.splitlines()]
        lines = [ln for ln in lines if ln]
        return "\n".join(lines)


def extract_url(url, fetcher):
    """`fetcher(url) -> (content_type, bytes)`, injected so server.py owns the actual network
    call (timeouts, size cap, redirect handling) and this module stays a pure text transform."""
    content_type, data = fetcher(url)
    ct = (content_type or "").split(";")[0].strip().lower()
    if ct in ("text/html", "application/xhtml+xml") or not ct:
        parser = _TextHTMLParser()
        try:
            parser.feed(data.decode("utf-8", errors="replace"))
        except Exception as e:
            raise RuntimeError(f"could not parse this page: {e}")
        text = parser.text()
        title = parser.title.strip()
    elif ct == "application/pdf":
        title = ""
        text = extract_pdf(data)
    elif ct.startswith("text/") or ct in ("application/json", "application/xml"):
        title = ""
        text = data.decode("utf-8", errors="replace")
    else:
        raise RuntimeError(f"unsupported content type for a URL: {ct or 'unknown'}")
    if not text.strip():
        raise RuntimeError("no extractable text found at this URL")
    body, truncated = _truncate(text)
    return title, body, truncated
