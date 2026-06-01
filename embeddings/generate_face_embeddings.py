"""Generate face embeddings for faculty images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGES = ROOT / "faculty_images"
DEFAULT_DOCS = ROOT / "processed" / "faculty_documents.json"
DEFAULT_OUTPUT = ROOT / "embeddings" / "face_embeddings.npy"
DEFAULT_META = ROOT / "embeddings" / "face_embeddings_meta.json"


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _find_image(images_dir: Path, faculty_id: str) -> Optional[Path]:
    patterns = [
        f"{faculty_id}.*",
    ]
    try:
        id_int = int(faculty_id)
        patterns.extend(
            [
                f"{id_int:03d}.*",
                f"{id_int:04d}.*",
                f"FAC{id_int:04d}.*",
            ]
        )
    except ValueError:
        pass

    for pattern in patterns:
        matches = sorted(images_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def generate_face_embeddings(
    images_dir: Path,
    documents_path: Path,
    output_path: Path,
    meta_path: Path,
    device: Optional[str] = None,
) -> np.ndarray:
    with documents_path.open("r", encoding="utf-8") as handle:
        documents: List[Dict[str, str]] = json.load(handle)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(image_size=160, margin=14, device=device)
    resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)

    embeddings: List[np.ndarray] = []
    meta: List[Dict[str, str]] = []

    for idx, doc in enumerate(documents):
        faculty_id = str(doc.get("id", "")).strip()
        if not faculty_id:
            continue
        image_path = _find_image(images_dir, faculty_id)
        if not image_path:
            continue
        try:
            image = Image.open(image_path).convert("RGB")
        except OSError:
            continue

        face = mtcnn(image)
        if face is None:
            continue

        with torch.no_grad():
            face_embedding = resnet(face.unsqueeze(0).to(device)).cpu().numpy()
        face_embedding = _normalize_rows(face_embedding.astype("float32"))[0]
        embeddings.append(face_embedding)
        meta.append(
            {
                "doc_index": idx,
                "id": faculty_id,
                "image_filename": image_path.name,
            }
        )

    if not embeddings:
        raise RuntimeError("No face embeddings were generated. Check image files and face detection.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    face_matrix = np.stack(embeddings, axis=0)
    np.save(output_path, face_matrix)
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, ensure_ascii=False)

    return face_matrix


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate face embeddings for faculty images")
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES, help="Path to faculty images")
    parser.add_argument("--documents", type=Path, default=DEFAULT_DOCS, help="Path to documents JSON")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Path to output face embeddings")
    parser.add_argument("--meta", type=Path, default=DEFAULT_META, help="Path to output metadata JSON")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()

    generate_face_embeddings(args.images_dir, args.documents, args.output, args.meta, args.device)
    print(f"Saved face embeddings to {args.output}")


if __name__ == "__main__":
    main()
