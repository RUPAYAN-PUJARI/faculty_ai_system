"""Face-based search using FAISS."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCUMENTS = ROOT / "processed" / "faculty_documents.json"
DEFAULT_INDEX = ROOT / "vector_db" / "faiss_face.index"
DEFAULT_META = ROOT / "embeddings" / "face_embeddings_meta.json"


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def load_documents(path: Path = DEFAULT_DOCUMENTS) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_face_index(path: Path = DEFAULT_INDEX) -> faiss.Index:
    return faiss.read_index(str(path))


def load_face_meta(path: Path = DEFAULT_META) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def create_face_models(device: Optional[str] = None) -> Tuple[MTCNN, InceptionResnetV1, str]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(image_size=160, margin=14, device=device)
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)
    return mtcnn, resnet, device


def embed_face(image: Image.Image, mtcnn: MTCNN, resnet: InceptionResnetV1, device: str) -> np.ndarray:
    face = mtcnn(image)
    if face is None:
        raise ValueError("No face detected in the uploaded image.")
    with torch.no_grad():
        embedding = resnet(face.unsqueeze(0).to(device)).cpu().numpy()
    return _normalize_rows(embedding.astype("float32"))


def search_face(
    image: Image.Image,
    documents: List[Dict[str, Any]],
    index: faiss.Index,
    meta: List[Dict[str, Any]],
    mtcnn: MTCNN,
    resnet: InceptionResnetV1,
    device: str,
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    query_emb = embed_face(image, mtcnn, resnet, device)
    search_k = min(len(meta), max(top_k * 5, top_k))
    scores, indices = index.search(query_emb, search_k)

    results: List[Dict[str, Any]] = []
    for rank, idx in enumerate(indices[0] if len(indices) else []):
        if idx < 0 or idx >= len(meta):
            continue
        meta_item = meta[idx]
        doc_index = meta_item.get("doc_index")
        if doc_index is None or doc_index >= len(documents):
            continue
        doc = dict(documents[doc_index])
        doc["rank"] = rank + 1
        doc["image_filename"] = meta_item.get("image_filename")
        results.append(doc)
        if len(results) >= top_k:
            break

    return results
