"""
vector_store.py - Modul Vector Database untuk Lexa Assistant Study Buddy
Menggunakan ChromaDB lokal dengan embedding open-source (HuggingFace) yang berjalan di CPU/GPU.
"""

import logging
import gc
from pathlib import Path
from typing import List, Optional, Tuple

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document  # fallback untuk versi lama

# ChromaDB via LangChain
try:
    from langchain_community.vectorstores import Chroma
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False

# HuggingFace Embeddings (open-source, offline, CPU/GPU)
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    _HF_EMBEDDINGS_AVAILABLE = True
except ImportError:
    _HF_EMBEDDINGS_AVAILABLE = False

logger = logging.getLogger(__name__)

# --- Konfigurasi Path ---
_MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _MODULE_DIR.parent.parent.parent
CHROMA_STORAGE_DIR = PROJECT_ROOT / "data" / "chroma_storage"

# --- Konfigurasi Model Embedding ---
# all-MiniLM-L6-v2: ringan (~80MB), sangat cepat, berjalan di CPU
# Alternatif lebih akurat: "BAAI/bge-small-en-v1.5"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Jumlah hasil similarity search default
DEFAULT_K_RESULTS = 3
SIMILARITY_SCORE_THRESHOLD = 0.5  # Skor minimal relevansi (0-1)


# ---------------------------------------------------------------------------
# Singleton: Embedding Model
# ---------------------------------------------------------------------------

_embedding_model: Optional[object] = None


def _get_embedding_model() -> Optional[object]:
    """
    Mengembalikan model embedding (lazy-loaded singleton).
    Model dimuat sekali dan digunakan kembali untuk efisiensi memori.
    """
    global _embedding_model

    if not _HF_EMBEDDINGS_AVAILABLE:
        logger.error(
            "HuggingFace Embeddings tidak tersedia. "
            "Install: pip install langchain-community sentence-transformers"
        )
        return None

    if _embedding_model is None:
        logger.info("Memuat model embedding: '%s' (CPU mode)...", EMBEDDING_MODEL_NAME)
        try:
            _embedding_model = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={"device": "cpu"},       # CPU untuk hemat VRAM GPU
                encode_kwargs={"normalize_embeddings": True},  # Normalisasi untuk cosine similarity
            )
            logger.info("Model embedding berhasil dimuat.")
        except Exception as e:
            logger.error("Gagal memuat model embedding: %s — %s", type(e).__name__, e)
            return None

    return _embedding_model


# ---------------------------------------------------------------------------
# Fungsi Utama
# ---------------------------------------------------------------------------

def _ensure_storage_dir() -> None:
    """Memastikan direktori penyimpanan ChromaDB sudah ada."""
    CHROMA_STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def get_vector_store() -> Optional[Chroma]:
    """
    Mengembalikan instance ChromaDB yang sudah ada (jika ada).
    Digunakan untuk similarity search tanpa perlu ingest ulang.

    Returns:
        Instance Chroma jika berhasil, None jika gagal atau belum ada data.
    """
    if not _CHROMA_AVAILABLE:
        logger.error("ChromaDB tidak tersedia. Install: pip install langchain-community chromadb")
        return None

    _ensure_storage_dir()
    embedding_model = _get_embedding_model()
    if embedding_model is None:
        return None

    try:
        vector_store = Chroma(
            persist_directory=str(CHROMA_STORAGE_DIR),
            embedding_function=embedding_model,
            collection_name="lexa_documents",
        )
        # Cek apakah collection sudah punya data
        count = vector_store._collection.count()
        if count == 0:
            logger.info("Vector store kosong. Belum ada dokumen yang diindeks.")
        else:
            logger.info("Vector store tersedia dengan %d chunk terindeks.", count)

        return vector_store

    except Exception as e:
        logger.error("Gagal membuka vector store: %s — %s", type(e).__name__, e)
        return None


def ingest_documents(chunks: List[Document]) -> Tuple[bool, str]:
    """
    Menyimpan chunk dokumen ke ChromaDB dengan embedding.
    Operasi ini akan menimpa koleksi yang ada (upsert).

    Args:
        chunks: List[Document] berisi chunk teks hasil splitting.

    Returns:
        Tuple (status_sukses: bool, pesan: str)
    """
    if not chunks:
        return (False, "Tidak ada chunk untuk diindeks. Periksa folder data/documents/.")

    if not _CHROMA_AVAILABLE:
        return (False, "ChromaDB tidak tersedia. Jalankan: pip install chromadb")

    _ensure_storage_dir()
    embedding_model = _get_embedding_model()
    if embedding_model is None:
        return (False, "Model embedding gagal dimuat. Periksa koneksi dan instalasi.")

    try:
        logger.info("Memulai ingest %d chunk ke ChromaDB...", len(chunks))

        # Buat atau timpa vector store dengan data baru
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=str(CHROMA_STORAGE_DIR),
            collection_name="lexa_documents",
        )

        final_count = vector_store._collection.count()
        msg = (
            f"Berhasil mengindeks {len(chunks)} chunk dari dokumen. "
            f"Total chunk dalam database: {final_count}."
        )
        logger.info(msg)
        return (True, msg)

    except Exception as e:
        msg = f"Gagal melakukan ingest dokumen ke ChromaDB: {type(e).__name__}: {e}"
        logger.error(msg)
        return (False, msg)
    finally:
        # Bersihkan variabel lokal untuk membantu GC
        gc.collect()


def similarity_search(
    query: str,
    k: int = DEFAULT_K_RESULTS,
    score_threshold: float = SIMILARITY_SCORE_THRESHOLD,
) -> List[Document]:
    """
    Mencari chunk dokumen yang paling relevan dengan query menggunakan similarity search.

    Args:
        query: Pertanyaan atau teks dari pengguna.
        k: Jumlah maksimum chunk yang dikembalikan.
        score_threshold: Skor minimal relevansi (0.0 - 1.0). Chunk di bawah threshold diabaikan.

    Returns:
        List[Document] yang relevan, diurutkan berdasarkan relevansi. Kosong jika tidak ada.
    """
    if not query or not isinstance(query, str) or len(query.strip()) < 3:
        logger.warning("Query similarity search tidak valid: %r", query)
        return []

    vector_store = get_vector_store()
    if vector_store is None:
        return []

    try:
        # Cek apakah ada data terindeks
        doc_count = vector_store._collection.count()
        if doc_count == 0:
            logger.info("Vector store kosong, skip similarity search.")
            return []

        logger.info("Menjalankan similarity search untuk query: '%s'", query[:80])

        # Gunakan similarity_search_with_relevance_scores untuk filter threshold
        results_with_scores = vector_store.similarity_search_with_relevance_scores(
            query=query.strip(),
            k=k,
        )

        # Filter berdasarkan threshold relevansi
        filtered_docs = [
            doc for doc, score in results_with_scores
            if score >= score_threshold
        ]

        if filtered_docs:
            logger.info(
                "Ditemukan %d chunk relevan (dari %d kandidat, threshold=%.2f).",
                len(filtered_docs), len(results_with_scores), score_threshold,
            )
        else:
            logger.info(
                "Tidak ada chunk yang melewati threshold %.2f. "
                "Skor tertinggi: %.4f.",
                score_threshold,
                max((s for _, s in results_with_scores), default=0.0),
            )

        return filtered_docs

    except Exception as e:
        logger.error("Error saat similarity search: %s — %s", type(e).__name__, e)
        return []
    finally:
        gc.collect()


def build_rag_context(query: str, k: int = DEFAULT_K_RESULTS) -> Optional[str]:
    """
    Membangun string konteks RAG dari hasil similarity search.
    Konteks ini siap disisipkan ke system prompt Ollama.

    Args:
        query: Pertanyaan dari pengguna.
        k: Jumlah chunk maksimum untuk diambil.

    Returns:
        String konteks RAG yang terformat, atau None jika tidak ada konteks relevan.
    """
    relevant_chunks = similarity_search(query=query, k=k)

    if not relevant_chunks:
        return None

    context_parts = []
    for i, doc in enumerate(relevant_chunks, start=1):
        source = doc.metadata.get("source_file", "Dokumen tidak diketahui")
        context_parts.append(
            f"[Sumber {i}: {source}]\n{doc.page_content.strip()}"
        )

    rag_context = "\n\n---\n\n".join(context_parts)
    logger.info("Konteks RAG dibangun dari %d chunk.", len(relevant_chunks))
    return rag_context


def get_collection_stats() -> dict:
    """
    Mengembalikan statistik koleksi ChromaDB.

    Returns:
        Dict berisi informasi koleksi.
    """
    try:
        vector_store = get_vector_store()
        if vector_store is None:
            return {"status": "error", "message": "Vector store tidak tersedia."}

        count = vector_store._collection.count()
        return {
            "status": "ok",
            "total_chunks": count,
            "storage_path": str(CHROMA_STORAGE_DIR),
            "embedding_model": EMBEDDING_MODEL_NAME,
            "collection_name": "lexa_documents",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
