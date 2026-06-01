"""Build a FAISS index from face embeddings."""

from __future__ import annotations

import argparse
from pathlib import Path

import faiss
import numpy as np


DEFAULT_EMBEDDINGS = Path(__file__).resolve().parents[1] / "embeddings" / "face_embeddings.npy"
DEFAULT_INDEX = Path(__file__).resolve().parents[1] / "vector_db" / "faiss_face.index"


def build_face_index(embeddings_path: Path, index_path: Path) -> faiss.Index:
	embeddings = np.load(embeddings_path).astype("float32")
	dimension = embeddings.shape[1]

	index = faiss.IndexFlatIP(dimension)
	index.add(embeddings)

	index_path.parent.mkdir(parents=True, exist_ok=True)
	faiss.write_index(index, str(index_path))
	return index


def main() -> None:
	parser = argparse.ArgumentParser(description="Build a FAISS index for face embeddings")
	parser.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS, help="Path to embeddings .npy")
	parser.add_argument("--output", type=Path, default=DEFAULT_INDEX, help="Path to output FAISS index")
	args = parser.parse_args()

	build_face_index(args.embeddings, args.output)
	print(f"Saved FAISS face index to {args.output}")


if __name__ == "__main__":
	main()
