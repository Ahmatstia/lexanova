"""
document_loader.py - Modul Study Buddy untuk Lexa Assistant
Memuat dan memproses dokumen PDF/Teks secara offline dari folder data/documents/.
"""

import logging
from pathlib import Path
from typing import List, Optional

# LangChain text processing (langchain v0.2+ API)
try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document  # fallback untuk versi lama

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter  # fallback

# PDF reader
try:
    from langchain_community.document_loaders import PyPDFLoader, TextLoader
    _LOADERS_AVAILABLE = True
except ImportError:
    _LOADERS_AVAILABLE = False

logger = logging.getLogger(__name__)

# --- Konfigurasi Path ---
_MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _MODULE_DIR.parent.parent.parent
DOCUMENTS_DIR = PROJECT_ROOT / "data" / "documents"

# --- Konfigurasi Text Splitter ---
CHUNK_SIZE = 500        # Ukuran chunk dalam karakter
CHUNK_OVERLAP = 50      # Overlap antar chunk agar konteks tidak terpotong
CHUNK_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]  # Urutan pemisah preferensi

# Ekstensi file yang didukung
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


# ---------------------------------------------------------------------------
# Text Splitter (singleton)
# ---------------------------------------------------------------------------

def _get_text_splitter() -> RecursiveCharacterTextSplitter:
    """Mengembalikan instance RecursiveCharacterTextSplitter yang dikonfigurasi."""
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )


# ---------------------------------------------------------------------------
# Fungsi Utama
# ---------------------------------------------------------------------------

def ensure_documents_dir() -> bool:
    """
    Memastikan folder documents/ sudah ada, membuatnya jika belum.

    Returns:
        True jika folder ada atau berhasil dibuat, False jika gagal.
    """
    try:
        DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Folder dokumen tersedia di: %s", DOCUMENTS_DIR)
        return True
    except OSError as e:
        logger.error("Gagal membuat folder dokumen '%s': %s", DOCUMENTS_DIR, e)
        return False


def load_single_document(file_path: Path) -> Optional[List[Document]]:
    """
    Memuat satu file dokumen (PDF atau TXT/MD) menjadi list Document LangChain.

    Args:
        file_path: Path absolut ke file dokumen.

    Returns:
        List[Document] jika berhasil dimuat, None jika gagal.
    """
    if not _LOADERS_AVAILABLE:
        logger.error("LangChain community loaders tidak tersedia. Install: langchain-community pypdf")
        return None

    if not file_path.exists():
        logger.warning("File tidak ditemukan: %s", file_path)
        return None

    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        logger.warning("Format file '%s' tidak didukung. Didukung: %s", suffix, SUPPORTED_EXTENSIONS)
        return None

    try:
        logger.info("Memuat dokumen: %s", file_path.name)

        if suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
        else:  # .txt atau .md
            loader = TextLoader(str(file_path), encoding="utf-8")

        documents = loader.load()

        # Tambahkan metadata sumber
        for doc in documents:
            doc.metadata["source_file"] = file_path.name
            doc.metadata["file_path"] = str(file_path)

        logger.info("Berhasil memuat %d halaman/bagian dari '%s'.", len(documents), file_path.name)
        return documents

    except UnicodeDecodeError:
        logger.warning("Gagal decode '%s' sebagai UTF-8, mencoba latin-1.", file_path.name)
        try:
            loader = TextLoader(str(file_path), encoding="latin-1")
            return loader.load()
        except Exception as e:
            logger.error("Gagal memuat '%s' dengan latin-1: %s", file_path.name, e)
            return None
    except Exception as e:
        logger.error("Error saat memuat dokumen '%s': %s — %s", file_path.name, type(e).__name__, e)
        return None


def load_all_documents(directory: Path = DOCUMENTS_DIR) -> List[Document]:
    """
    Memuat semua dokumen yang didukung dari sebuah direktori.

    Args:
        directory: Path ke direktori dokumen. Default: DOCUMENTS_DIR.

    Returns:
        List[Document] dari semua file yang berhasil dimuat.
    """
    ensure_documents_dir()

    if not directory.exists():
        logger.error("Direktori '%s' tidak ditemukan.", directory)
        return []

    all_documents: List[Document] = []
    found_files = [
        f for f in directory.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not found_files:
        logger.warning(
            "Tidak ada dokumen yang ditemukan di '%s'. "
            "Tambahkan file PDF/TXT/MD ke folder data/documents/.",
            directory,
        )
        return []

    logger.info("Ditemukan %d file dokumen untuk dimuat.", len(found_files))

    for file_path in found_files:
        docs = load_single_document(file_path)
        if docs:
            all_documents.extend(docs)

    logger.info("Total %d halaman/bagian dimuat dari %d file.", len(all_documents), len(found_files))
    return all_documents


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Memecah dokumen menjadi chunk-chunk kecil menggunakan RecursiveCharacterTextSplitter.

    Args:
        documents: List[Document] yang akan dipecah.

    Returns:
        List[Document] berisi chunk-chunk hasil splitting.
    """
    if not documents:
        logger.warning("Tidak ada dokumen untuk di-split.")
        return []

    try:
        splitter = _get_text_splitter()
        chunks = splitter.split_documents(documents)

        # Tambahkan metadata chunk index ke setiap chunk
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)

        logger.info(
            "Splitting selesai: %d dokumen -> %d chunk (ukuran: %d, overlap: %d).",
            len(documents), len(chunks), CHUNK_SIZE, CHUNK_OVERLAP,
        )
        return chunks

    except Exception as e:
        logger.error("Error saat splitting dokumen: %s — %s", type(e).__name__, e)
        return []


def load_and_split_documents(directory: Path = DOCUMENTS_DIR) -> List[Document]:
    """
    Fungsi convenience: memuat semua dokumen dari direktori dan langsung split.

    Args:
        directory: Path ke direktori dokumen.

    Returns:
        List[Document] berisi chunk siap indeks ke vector store.
    """
    raw_documents = load_all_documents(directory)
    if not raw_documents:
        return []
    return split_documents(raw_documents)


def list_available_documents(directory: Path = DOCUMENTS_DIR) -> List[dict]:
    """
    Mengembalikan informasi tentang dokumen yang tersedia di direktori.

    Returns:
        List dict dengan info file (nama, ukuran, format).
    """
    ensure_documents_dir()

    result = []
    if not directory.exists():
        return result

    for f in directory.iterdir():
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
            result.append({
                "name": f.name,
                "size_kb": round(f.stat().st_size / 1024, 2),
                "format": f.suffix.lower().lstrip(".").upper(),
                "path": str(f),
            })

    return sorted(result, key=lambda x: x["name"])
