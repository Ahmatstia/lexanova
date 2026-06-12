"""Lexa Assistant - Study Buddy (RAG) Module"""
# Import di-lazy agar tidak memblock startup server
# Gunakan import langsung dari submodule ketika dibutuhkan

__all__ = [
    "load_and_split_documents",
    "load_all_documents",
    "list_available_documents",
    "ingest_documents",
    "similarity_search",
    "build_rag_context",
    "get_collection_stats",
]
