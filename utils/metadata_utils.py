"""Metadata artifact utilities for embeddings and vector DB."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List


def build_metadata_artifacts(documents_path: Path, ids_path: Path, metadata_path: Path, metadata_json: Path) -> None:
    with documents_path.open("r", encoding="utf-8") as handle:
        documents: List[Dict[str, Any]] = json.load(handle)

    ids = [doc.get("id") for doc in documents]
    metadata = [
        {
            "id": doc.get("id"),
            "name": doc.get("name"),
            "department": doc.get("department"),
            "present_designation": doc.get("present_designation"),
            "photo_path": doc.get("photo_path"),
        }
        for doc in documents
    ]

    ids_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_json.parent.mkdir(parents=True, exist_ok=True)

    with ids_path.open("wb") as handle:
        pickle.dump(ids, handle)

    with metadata_path.open("wb") as handle:
        pickle.dump(metadata, handle)

    with metadata_json.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
