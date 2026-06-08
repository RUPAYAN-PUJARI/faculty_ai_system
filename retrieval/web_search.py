"""Scrape and search BPPIMT website content for admissions, timetable, and facilities."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import faiss
import numpy as np
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
WEB_DOCS_PATH = ROOT / "processed" / "web_documents.json"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

WEB_SEED_URLS = [
    "https://bppimt.ac.in/admission/",
    "https://bppimt.ac.in/admission/under-graduate-courses-b-tech/",
    "https://bppimt.ac.in/admission/other-under-graduate-courses/",
    "https://bppimt.ac.in/admission/post-graduate-course-m-tech/",
    "https://bppimt.ac.in/admission/post-graduate-course-mca/",
    "https://bppimt.ac.in/admission/scholarship-schemes/",
    "https://bppimt.ac.in/students/routine/",
    "https://bppimt.ac.in/students/institute-calendar/",
    "https://bppimt.ac.in/students/list-of-holidays/",
    "https://bppimt.ac.in/central-facilities/",
    "https://bppimt.ac.in/central-facilities/computer-facility/",
    "https://bppimt.ac.in/central-facilities/language-laboratory/",
    "https://bppimt.ac.in/central-facilities/sports-and-games-facilities/",
    "https://bppimt.ac.in/institute-cells/central-library/",
    "https://bppimt.ac.in/institute-cells/hostel-welfare-committee/",
    "https://bppimt.ac.in/contact-us/",
]

RELEVANT_PATH_KEYWORDS = {
    "admission",
    "students",
    "routine",
    "timetable",
    "calendar",
    "holiday",
    "central-facilities",
    "facility",
    "library",
    "hostel",
    "computer-facility",
    "language-laboratory",
    "sports-and-games",
    "scholarship",
    "fees",
    "contact",
}

INTENT_FILTERS: list[tuple[tuple[str, ...], set[str]]] = [
    (
        ("admission", "admissions", "apply", "application", "eligibility", "intake"),
        {"admission", "admissions", "eligibility", "apply"},
    ),
    (("fee", "fees", "tuition", "payment"), {"fee", "fees", "tuition"}),
    (
        ("timetable", "time table", "routine", "schedule", "class schedule", "class routine"),
        {"routine", "timetable", "time table"},
    ),
    (("calendar", "academic calendar", "institute calendar"), {"calendar", "institute calendar"}),
    (("holiday", "holidays", "holiday list", "list of holidays"), {"holiday", "holidays", "list of holidays"}),
    (("library", "lending"), {"library", "central library"}),
    (("hostel",), {"hostel", "boys hostel", "girls hostel"}),
    (
        ("facility", "facilities", "lab", "laboratory", "sports", "canteen", "medical"),
        {
            "facility",
            "facilities",
            "central facilities",
            "laboratory",
            "lab",
            "sports",
            "computer facility",
            "language laboratory",
            "canteen",
            "medical",
        },
    ),
    (
        ("course", "courses", "program", "programs", "b.tech", "m.tech", "mca", "bba", "bca"),
        {"course", "courses", "program", "programs", "b.tech", "m.tech", "mca", "bba", "bca"},
    ),
    (("contact", "phone", "email", "address", "location"), {"contact", "phone", "email", "address"}),
    (("scholarship", "scheme", "schemes"), {"scholarship", "scheme", "schemes"}),
]

URL_PATH_HINTS: dict[str, list[str]] = {
    "holiday": ["holiday", "holidays"],
    "holidays": ["holiday", "holidays"],
    "holiday list": ["holiday"],
    "list of holidays": ["holiday"],
    "routine": ["routine"],
    "timetable": ["routine", "timetable"],
    "calendar": ["institute-calendar", "calendar"],
    "academic calendar": ["institute-calendar"],
    "admission": ["admission"],
    "hostel": ["hostel"],
    "library": ["library"],
    "sports": ["sports"],
    "scholarship": ["scholarship"],
    "contact": ["contact"],
}


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _clean_text(text: str) -> str:
    return " ".join(text.split())


def _intent_terms_for_query(query: str) -> set[str]:
    query_lower = query.lower()
    terms: set[str] = set()
    for triggers, intent_terms in INTENT_FILTERS:
        if any(trigger in query_lower for trigger in triggers):
            terms.update(intent_terms)
    return terms


def _url_path_hints_for_query(query: str) -> list[str]:
    """Return URL path segments that strongly signal relevance for this query."""
    query_lower = query.lower()
    hints: list[str] = []
    for trigger, segments in URL_PATH_HINTS.items():
        if trigger in query_lower:
            hints.extend(segments)
    return list(dict.fromkeys(hints))  


def _doc_matches_url_hints(doc: Dict[str, Any], url_hints: list[str]) -> bool:
    if not url_hints:
        return False
    url_path = urlparse(doc.get("url", "")).path.lower()
    return any(hint in url_path for hint in url_hints)


def _doc_matches_terms(doc: Dict[str, Any], terms: set[str], include_content: bool = True) -> bool:
    if not terms:
        return True
    parts = [str(doc.get("title", "")), str(doc.get("url", ""))]
    if include_content:
        parts.append(str(doc.get("content", "")))
    haystack = " ".join(parts).lower()
    return any(term in haystack for term in terms)


def _title_url_relevance(doc: Dict[str, Any], query_tokens: list[str]) -> int:
    """Count how many query tokens appear in title or URL path (for re-ranking)."""
    url_path = urlparse(doc.get("url", "")).path.lower().replace("-", " ").replace("/", " ")
    title = doc.get("title", "").lower()
    return sum(1 for t in query_tokens if len(t) > 2 and (t in url_path or t in title))


def _is_relevant_link(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    if "bppimt.ac.in" not in parsed.netloc:
        return False
    path = parsed.path.lower()
    return any(keyword in path for keyword in RELEVANT_PATH_KEYWORDS)


def _fetch_url(url: str, timeout: int = 20) -> Optional[str]:
    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0 (BPPIMT Bot)"},
        )
        if response.status_code >= 400:
            return None
        return response.text
    except requests.RequestException:
        return None


def _extract_page_data(html: str, base_url: str) -> Tuple[str, str, List[Dict[str, str]], List[str]]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()

    title = ""
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        title = h1.get_text(strip=True)
    elif soup.title and soup.title.get_text(strip=True):
        title = soup.title.get_text(strip=True)
    if not title:
        title = base_url

    text = _clean_text(soup.get_text(separator=" "))

    links: List[str] = []
    pdf_links: List[Dict[str, str]] = []
    for anchor in soup.find_all("a", href=True):
        href = anchor.get("href", "").strip()
        if not href:
            continue
        absolute = urljoin(base_url, href)
        absolute = absolute.split("#")[0]
        if absolute.lower().endswith(".pdf"):
            label = _clean_text(anchor.get_text(" ", strip=True)) or "PDF"
            pdf_links.append({"title": label, "url": absolute})
        if _is_relevant_link(absolute):
            links.append(absolute)

    return title, text, pdf_links, links


def build_web_documents(
    seed_urls: Iterable[str] = WEB_SEED_URLS,
    max_pages: int = 24,
    max_chars: int = 9000,
) -> List[Dict[str, Any]]:
    documents: List[Dict[str, Any]] = []
    visited: set[str] = set()
    queue: List[str] = list(seed_urls)

    while queue and len(visited) < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)
        html = _fetch_url(url)
        if not html:
            continue

        title, text, pdf_links, links = _extract_page_data(html, url)
        if not text:
            continue

        trimmed_text = text[:max_chars]
        if len(text) > max_chars:
            trimmed_text = f"{trimmed_text}..."

        documents.append(
            {
                "id": f"web-{len(documents) + 1}",
                "title": title,
                "url": url,
                "content": trimmed_text,
                "pdf_links": pdf_links,
                "source": "bppimt_web",
            }
        )

        for link in links:
            if link not in visited and link not in queue and len(visited) + len(queue) < max_pages:
                queue.append(link)

    return documents


def save_web_documents(documents: List[Dict[str, Any]], path: Path = WEB_DOCS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(documents, handle, ensure_ascii=False, indent=2)


def load_web_documents(path: Path = WEB_DOCS_PATH) -> Optional[List[Dict[str, Any]]]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_web_index(
    documents: List[Dict[str, Any]],
    model_name: str = DEFAULT_MODEL,
) -> Tuple[Optional[faiss.Index], Optional[np.ndarray]]:
    if not documents:
        return None, None
    model = SentenceTransformer(model_name)
    texts = [doc.get("content", "") for doc in documents]
    embeddings = model.encode(texts, convert_to_numpy=True)
    embeddings = _normalize_rows(embeddings.astype("float32"))
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings


def _build_snippet(text: str, query: str, max_chars: int = 260) -> str:
    if not text:
        return ""
    normalized_text = text.replace("\n", " ")
    if not query:
        return normalized_text[:max_chars].strip()
    terms = [term for term in re.findall(r"[a-zA-Z0-9]+", query.lower()) if len(term) > 2]
    lower_text = normalized_text.lower()
    for term in terms:
        idx = lower_text.find(term)
        if idx != -1:
            start = max(0, idx - 100)
            end = min(len(normalized_text), idx + 160)
            snippet = normalized_text[start:end].strip()
            if start > 0:
                snippet = f"...{snippet}"
            if end < len(normalized_text):
                snippet = f"{snippet}..."
            return snippet
    return normalized_text[:max_chars].strip()


def search_web_documents(
    query: str,
    documents: List[Dict[str, Any]],
    index: Optional[faiss.Index],
    model_name: str = DEFAULT_MODEL,
    top_k: int = 6,
) -> List[Dict[str, Any]]:
    if not documents:
        return []

    intent_terms = _intent_terms_for_query(query)
    url_hints = _url_path_hints_for_query(query)
    query_tokens = [token for token in re.findall(r"[a-zA-Z0-9]+", query.lower()) if token]
    strict_match = len(query_tokens) <= 4

    if not query.strip() or index is None:
        candidates = documents[:]
    else:
        model = SentenceTransformer(model_name)
        query_emb = model.encode([query], convert_to_numpy=True)
        query_emb = _normalize_rows(query_emb.astype("float32"))
        scores, indices = index.search(query_emb, min(max(top_k * 3, 12), len(documents)))
        candidates = [documents[idx] for idx in indices[0] if idx >= 0]

    if url_hints:
        url_matched = [doc for doc in candidates if _doc_matches_url_hints(doc, url_hints)]
        if not url_matched:
            url_matched = [doc for doc in documents if _doc_matches_url_hints(doc, url_hints)]
        if url_matched:
            results = url_matched[:top_k]
            enriched: List[Dict[str, Any]] = []
            for doc in results:
                item = dict(doc)
                item["snippet"] = _build_snippet(doc.get("content", ""), query)
                enriched.append(item)
            return enriched

    if intent_terms:
        filtered = [
            doc for doc in candidates
            if _doc_matches_terms(doc, intent_terms, include_content=not strict_match)
        ]
        if not filtered:
            filtered = [
                doc for doc in documents
                if _doc_matches_terms(doc, intent_terms, include_content=False)
            ]
        if filtered:
            filtered.sort(key=lambda d: _title_url_relevance(d, query_tokens), reverse=True)
            candidates = filtered

    results = candidates[:top_k]
    enriched = []
    for doc in results:
        item = dict(doc)
        item["snippet"] = _build_snippet(doc.get("content", ""), query)
        enriched.append(item)
    return enriched


class WebIndexCache:
    def __init__(self) -> None:
        self.documents: List[Dict[str, Any]] = []
        self.index: Optional[faiss.Index] = None
        self.last_loaded: float = 0.0

    def refresh(
        self,
        ttl_seconds: int = 21600,
        model_name: str = DEFAULT_MODEL,
        seed_urls: Iterable[str] = WEB_SEED_URLS,
    ) -> None:
        now = time.time()
        if self.documents and self.index is not None and now - self.last_loaded < ttl_seconds:
            return

        documents = load_web_documents() or build_web_documents(seed_urls=seed_urls)
        if documents:
            save_web_documents(documents)
        index, _embeddings = build_web_index(documents, model_name=model_name)
        self.documents = documents
        self.index = index
        self.last_loaded = now
