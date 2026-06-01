"""Generate embeddings for faculty documents."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

import numpy as np

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")

from sentence_transformers import SentenceTransformer


DEFAULT_DOCUMENTS = Path(__file__).resolve().parents[1] / "processed" / "faculty_documents.json"
DEFAULT_EMBEDDINGS = Path(__file__).resolve().parents[1] / "embeddings" / "text_embeddings.npy"
DEFAULT_META = Path(__file__).resolve().parents[1] / "embeddings" / "text_embeddings_meta.json"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
	norms = np.linalg.norm(matrix, axis=1, keepdims=True)
	norms[norms == 0] = 1.0
	return matrix / norms


def generate_embeddings(documents_path: Path, output_path: Path, meta_path: Path, model_name: str) -> np.ndarray:
	with documents_path.open("r", encoding="utf-8") as handle:
		documents = json.load(handle)

	texts: List[str] = [doc.get("text", "") for doc in documents]
	model = SentenceTransformer(model_name)
	embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
	embeddings = _normalize_rows(embeddings.astype("float32"))

	output_path.parent.mkdir(parents=True, exist_ok=True)
	np.save(output_path, embeddings)

	meta = {
		"model": model_name,
		"count": len(texts),
		"ids": [doc.get("id") for doc in documents],
	}
	with meta_path.open("w", encoding="utf-8") as handle:
		json.dump(meta, handle, indent=2, ensure_ascii=False)

	return embeddings


def main() -> None:
	parser = argparse.ArgumentParser(description="Generate embeddings for faculty documents")
	parser.add_argument("--documents", type=Path, default=DEFAULT_DOCUMENTS, help="Path to documents JSON")
	parser.add_argument("--output", type=Path, default=DEFAULT_EMBEDDINGS, help="Path to output .npy file")
	parser.add_argument("--meta", type=Path, default=DEFAULT_META, help="Path to output metadata JSON")
	parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="SentenceTransformer model name")
	args = parser.parse_args()

	generate_embeddings(args.documents, args.output, args.meta, args.model)
	print(f"Saved embeddings to {args.output}")


if __name__ == "__main__":
	main()
