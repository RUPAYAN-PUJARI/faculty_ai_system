# Faculty AI System

This workspace contains a text-based retrieval pipeline for faculty profiles, with placeholders for future image-based search and a full stack application.

## Current text pipeline

- `embeddings/generate_text_documents.py` builds JSON documents from the CSV.
- `embeddings/generate_text_embeddings.py` creates sentence embeddings.
- `vector_db/build_faiss_index.py` builds the FAISS index.
- `retrieval/text_search.py` performs text-based retrieval.

## Planned extensions

- Face embeddings and face search
- Hybrid search and reranking
- API backend + web/mobile frontends
