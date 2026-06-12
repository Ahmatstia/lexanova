"""Lexa Assistant - Study Buddy (RAG) Module"""
from app.modules.study_buddy.document_loader import (
    load_and_split_documents,
    load_all_documents,
    list_available_documents,
)
from app.modules.study_buddy.vector_store import (
    ingest_documents,
    similarity_search,
    build_rag_context,
    get_collection_stats,
)

__all__ = [
    "load_and_split_documents",
    "load_all_documents",
    "list_available_documents",
    "ingest_documents",
    "similarity_search",
    "build_rag_context",
    "get_collection_stats",
]
