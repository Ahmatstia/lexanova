"""
config.py - Konfigurasi terpusat Lexa Assistant
Membaca semua nilai dari environment variables (.env) dengan fallback defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Lokasi root project (dideteksi secara dinamis dari posisi file ini)
# config.py berada di app/core/, jadi naik 2 level = root project
_CONFIG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _CONFIG_DIR.parent.parent

# Muat file .env dari root project
_ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=_ENV_PATH)


class Settings:
    """Konfigurasi aplikasi yang dibaca dari environment variables."""

    # --- Ollama ---
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")
    OLLAMA_TIMEOUT_SEC: int = int(os.getenv("OLLAMA_TIMEOUT_SEC", "120"))

    # --- Paths (dinamis, bukan hardcoded) ---
    PROJECT_ROOT: Path = PROJECT_ROOT
    DOCUMENTS_DIR: Path = PROJECT_ROOT / "data" / "documents"
    CHROMA_STORAGE_DIR: Path = PROJECT_ROOT / "data" / "chroma_storage"

    # Override path dari env jika disediakan
    _docs_override = os.getenv("DOCUMENTS_DIR")
    if _docs_override:
        DOCUMENTS_DIR = Path(_docs_override)

    _chroma_override = os.getenv("CHROMA_STORAGE_DIR")
    if _chroma_override:
        CHROMA_STORAGE_DIR = Path(_chroma_override)

    # --- Study Buddy ---
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    RAG_K_RESULTS: int = int(os.getenv("RAG_K_RESULTS", "3"))
    RAG_SCORE_THRESHOLD: float = float(os.getenv("RAG_SCORE_THRESHOLD", "0.5"))

    # --- Monitor ---
    DEFAULT_CAMERA_INDEX: int = int(os.getenv("DEFAULT_CAMERA_INDEX", "0"))

    def __repr__(self) -> str:
        return (
            f"Settings("
            f"ollama_url={self.OLLAMA_BASE_URL!r}, "
            f"model={self.OLLAMA_MODEL!r}, "
            f"project_root={self.PROJECT_ROOT})"
        )


settings = Settings()