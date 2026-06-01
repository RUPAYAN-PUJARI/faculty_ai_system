"""Project entrypoint for building and querying the faculty search system."""

from __future__ import annotations

import argparse
from pathlib import Path

from embeddings.generate_text_documents import generate_documents
from embeddings.generate_placement_documents import generate_documents as generate_placement_documents
from embeddings.generate_scholarship_documents import generate_documents as generate_scholarship_documents
from embeddings.generate_text_embeddings import generate_embeddings
from embeddings.generate_face_embeddings import generate_face_embeddings
from retrieval.text_search import search
from utils.metadata_utils import build_metadata_artifacts
from vector_db.build_faiss_index import build_index
from vector_db.build_faiss_face_index import build_face_index


ROOT = Path(__file__).resolve().parent


def build_all(include_faces: bool = True) -> None:
    documents_path = ROOT / "processed" / "faculty_documents.json"
    generate_documents(ROOT / "data" / "faculty_list_cleaned.csv", documents_path)
    generate_embeddings(
        documents_path,
        ROOT / "embeddings" / "text_embeddings.npy",
        ROOT / "embeddings" / "text_embeddings_meta.json",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    build_index(ROOT / "embeddings" / "text_embeddings.npy", ROOT / "vector_db" / "faiss_text.index")
    build_metadata_artifacts(
        documents_path,
        ROOT / "embeddings" / "faculty_ids.pkl",
        ROOT / "vector_db" / "metadata.pkl",
        ROOT / "data" / "faculty_metadata.json",
    )

    placement_documents_path = ROOT / "processed" / "placement_documents.json"
    generate_placement_documents(ROOT / "data" / "placement_details.csv", placement_documents_path)
    generate_embeddings(
        placement_documents_path,
        ROOT / "embeddings" / "placement_embeddings.npy",
        ROOT / "embeddings" / "placement_embeddings_meta.json",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    build_index(
        ROOT / "embeddings" / "placement_embeddings.npy",
        ROOT / "vector_db" / "faiss_placement.index",
    )

    scholarship_documents_path = ROOT / "processed" / "scholarship_documents.json"
    generate_scholarship_documents(ROOT / "data" / "scholarship_details.json", scholarship_documents_path)
    generate_embeddings(
        scholarship_documents_path,
        ROOT / "embeddings" / "scholarship_embeddings.npy",
        ROOT / "embeddings" / "scholarship_embeddings_meta.json",
        "sentence-transformers/all-MiniLM-L6-v2",
    )
    build_index(
        ROOT / "embeddings" / "scholarship_embeddings.npy",
        ROOT / "vector_db" / "faiss_scholarship.index",
    )
    if include_faces:
        images_dir = ROOT / "faculty_images"
        if images_dir.exists():
            generate_face_embeddings(
                images_dir,
                documents_path,
                ROOT / "embeddings" / "face_embeddings.npy",
                ROOT / "embeddings" / "face_embeddings_meta.json",
            )
            build_face_index(ROOT / "embeddings" / "face_embeddings.npy", ROOT / "vector_db" / "faiss_face.index")


def run_search(query: str, top_k: int) -> None:
    from retrieval.text_search import _load_documents, _load_index

    documents = _load_documents(ROOT / "processed" / "faculty_documents.json")
    index = _load_index(ROOT / "vector_db" / "faiss_text.index")
    results = search(query, documents, index, "sentence-transformers/all-MiniLM-L6-v2", top_k=top_k)
    for item in results:
        print(f"{item.get('rank')}. {item.get('name')} — {item.get('present_designation')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Faculty AI System utilities")
    sub = parser.add_subparsers(dest="command", required=True)

    build_cmd = sub.add_parser("build", help="Generate documents, embeddings, and indexes")
    build_cmd.add_argument("--skip-faces", action="store_true", help="Skip face embedding/index build")
    build_cmd.set_defaults(func=lambda args: build_all(not args.skip_faces))

    faces_cmd = sub.add_parser("build-faces", help="Generate face embeddings and index only")
    faces_cmd.set_defaults(
        func=lambda _args: (
            generate_face_embeddings(
                ROOT / "faculty_images",
                ROOT / "processed" / "faculty_documents.json",
                ROOT / "embeddings" / "face_embeddings.npy",
                ROOT / "embeddings" / "face_embeddings_meta.json",
            ),
            build_face_index(
                ROOT / "embeddings" / "face_embeddings.npy",
                ROOT / "vector_db" / "faiss_face.index",
            ),
        )
    )

    search_cmd = sub.add_parser("search", help="Run a quick search")
    search_cmd.add_argument("--query", required=True, help="Search query")
    search_cmd.add_argument("--top-k", type=int, default=5, help="Number of results")
    search_cmd.set_defaults(func=lambda args: run_search(args.query, args.top_k))

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
