"""Search faculty documents using text embeddings and FAISS."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import re

import faiss
import numpy as np

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCUMENTS = ROOT / "processed" / "faculty_documents.json"
DEFAULT_INDEX = ROOT / "vector_db" / "faiss_text.index"
DEFAULT_EMBEDDINGS_META = ROOT / "embeddings" / "text_embeddings_meta.json"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
	norms = np.linalg.norm(matrix, axis=1, keepdims=True)
	norms[norms == 0] = 1.0
	return matrix / norms


def _load_documents(path: Path) -> List[Dict[str, Any]]:
	with path.open("r", encoding="utf-8") as handle:
		return json.load(handle)


def _load_index(path: Path) -> faiss.Index:
	return faiss.read_index(str(path))


def _normalize_value(value: Optional[str]) -> str:
	return (value or "").strip().lower()


def _normalize_name(value: Optional[str]) -> str:
	text = _normalize_value(value)
	if not text:
		return ""
	text = re.sub(r"\b(mr|mrs|ms|dr|prof|professor)\.?\b", "", text, flags=re.IGNORECASE)
	text = re.sub(r"[^a-z\s]", " ", text)
	text = " ".join(text.split())
	return text


def _filter_doc(doc: Dict[str, Any], name: Optional[str], designation: Optional[str], department: Optional[str]) -> bool:
	name_value = _normalize_name(name)
	designation_value = _normalize_value(designation)
	department_value = _normalize_value(department)

	if name_value and name_value not in _normalize_name(doc.get("name")):
		return False
	if designation_value and designation_value != _normalize_value(doc.get("present_designation")):
		return False
	if department_value and department_value not in _normalize_value(doc.get("department")):
		return False
	return True


def search_text_documents(
	query: str,
	documents: List[Dict[str, Any]],
	index: Optional[faiss.Index],
	model_name: str,
	top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
	query_value = query.strip()
	if not documents:
		return []
	if not query_value or index is None:
		results = documents if top_k is None else documents[:top_k]
		for idx, doc in enumerate(results):
			doc["rank"] = idx + 1
		return results

	model = SentenceTransformer(model_name)
	query_emb = model.encode([query_value], convert_to_numpy=True)
	query_emb = _normalize_rows(query_emb.astype("float32"))

	effective_top_k = top_k or 5
	search_k = min(len(documents), max(effective_top_k * 5, effective_top_k))
	scores, indices = index.search(query_emb, search_k)
	candidates = [documents[i] for i in indices[0] if i >= 0]

	results = candidates if top_k is None else candidates[:top_k]
	for idx, doc in enumerate(results):
		doc["rank"] = idx + 1
	return results


def search(
	query: str,
	documents: List[Dict[str, Any]],
	index: faiss.Index,
	model_name: str,
	top_k: Optional[int] = None,
	name: Optional[str] = None,
	designation: Optional[str] = None,
	department: Optional[str] = None,
) -> List[Dict[str, Any]]:
	query_value = query.strip()
	if not query_value:
		candidates = [doc for doc in documents if _filter_doc(doc, name, designation, department)]
		results = candidates if top_k is None else candidates[:top_k]
		for idx, doc in enumerate(results):
			doc["rank"] = idx + 1
		return results

	model = SentenceTransformer(model_name)
	query_emb = model.encode([query_value], convert_to_numpy=True)
	query_emb = _normalize_rows(query_emb.astype("float32"))

	effective_top_k = top_k or 5
	search_k = min(len(documents), max(effective_top_k * 5, effective_top_k))
	scores, indices = index.search(query_emb, search_k)
	candidates = [documents[i] for i in indices[0] if i >= 0]

	if any([name, designation, department]):
		filtered = [doc for doc in candidates if _filter_doc(doc, name, designation, department)]
		if filtered:
			candidates = filtered

	results = candidates if top_k is None else candidates[:top_k]
	for idx, doc in enumerate(results):
		doc["rank"] = idx + 1
	return results


def main() -> None:
	parser = argparse.ArgumentParser(description="Search faculty profiles using FAISS and embeddings")
	parser.add_argument("--query", type=str, required=True, help="Search query text")
	parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
	parser.add_argument("--documents", type=Path, default=DEFAULT_DOCUMENTS, help="Path to documents JSON")
	parser.add_argument("--index", type=Path, default=DEFAULT_INDEX, help="Path to FAISS index")
	parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="SentenceTransformer model name")
	parser.add_argument("--name", type=str, default=None, help="Filter by name contains")
	parser.add_argument("--designation", type=str, default=None, help="Filter by designation contains")
	parser.add_argument("--department", type=str, default=None, help="Filter by department contains")
	args = parser.parse_args()

	documents = _load_documents(args.documents)
	index = _load_index(args.index)

	results = search(
		query=args.query,
		documents=documents,
		index=index,
		model_name=args.model,
		top_k=args.top_k,
		name=args.name,
		designation=args.designation,
		department=args.department,
	)

	print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
	main()
