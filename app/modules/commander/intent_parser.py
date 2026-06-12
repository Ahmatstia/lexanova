"""
intent_parser.py - Parser Intent untuk Lexa Assistant OS-Commander
Mendeteksi variasi perintah pengguna menggunakan regex dan tokenisasi fleksibel.
Jika eksekusi OS gagal, kode error dikembalikan agar dapat dikirim ke Ollama sebagai konteks.
"""

import re
import logging
from dataclasses import dataclass
from typing import Callable, Optional

from app.modules.commander.os_actions import (
    open_vs_code,
    open_local_folder,
    open_browser_tab,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class IntentResult:
    """Hasil dari proses parsing dan eksekusi intent."""
    detected: bool          # True jika intent OS berhasil dideteksi
    success: bool           # True jika eksekusi OS berhasil
    message: str            # Pesan sukses atau keterangan error
    error_context: str = "" # Konteks error untuk dikirim ke Ollama jika gagal


# ---------------------------------------------------------------------------
# Pola Regex untuk Deteksi Intent
# ---------------------------------------------------------------------------

# Kata kerja tindakan yang berarti "buka/jalankan"
_VERB_PATTERN = r"(?:buka|open|jalankan|launch|tampilkan|show|akses|start)"

# Pola intent: membuka VS Code
_VSCODE_PATTERN = re.compile(
    rf"{_VERB_PATTERN}\s+(?:vs\s?code|vscode|visual\s+studio\s+code|editor\s+kode)",
    re.IGNORECASE,
)

# Pola intent: membuka folder
_FOLDER_PATTERN = re.compile(
    rf"{_VERB_PATTERN}\s+(?:folder|direktori|directory|dir)\s+(?P<folder_name>\w[\w\s\-]*)",
    re.IGNORECASE,
)

# Pola intent: membuka browser/website
_BROWSER_PATTERN = re.compile(
    rf"{_VERB_PATTERN}\s+(?:browser|web|website|situs|tab|halaman)?\s*(?P<site_name>[\w\.\-:/]+)",
    re.IGNORECASE,
)

# Alias normalisasi nama folder (memetakan variasi bahasa ke key standar)
_FOLDER_ALIASES: dict[str, str] = {
    "skripsi": "skripsi",
    "tugas": "skripsi",
    "thesis": "skripsi",
    "dokumen": "skripsi",
    "document": "skripsi",
    "download": "download",
    "unduhan": "download",
    "project": "project",
    "proyek": "project",
    "data": "data",
}

# Alias normalisasi nama website
_SITE_ALIASES: dict[str, str] = {
    "github": "github",
    "git": "github",
    "chatgpt": "chatgpt",
    "gpt": "chatgpt",
    "openai": "chatgpt",
    "gemini": "gemini",
    "google gemini": "gemini",
    "youtube": "youtube",
    "yt": "youtube",
    "google": "google",
}


# ---------------------------------------------------------------------------
# Helper: Normalisasi Alias
# ---------------------------------------------------------------------------

def _normalize_folder_name(raw_name: str) -> Optional[str]:
    """Mencocokkan nama folder mentah ke key standar."""
    raw_lower = raw_name.strip().lower()
    # Exact match dahulu
    if raw_lower in _FOLDER_ALIASES:
        return _FOLDER_ALIASES[raw_lower]
    # Partial match (substring)
    for alias, key in _FOLDER_ALIASES.items():
        if alias in raw_lower:
            return key
    return None


def _normalize_site_name(raw_name: str) -> Optional[str]:
    """Mencocokkan nama site mentah ke key standar atau URL langsung."""
    raw_lower = raw_name.strip().lower()
    if raw_lower in _SITE_ALIASES:
        return _SITE_ALIASES[raw_lower]
    for alias, key in _SITE_ALIASES.items():
        if alias in raw_lower:
            return key
    # Jika ada URL https:// langsung, kembalikan as-is
    if raw_lower.startswith("https://"):
        return raw_name.strip()
    return None


# ---------------------------------------------------------------------------
# Parser Utama
# ---------------------------------------------------------------------------

def parse_and_execute_intent(user_message: str) -> Optional[IntentResult]:
    """
    Menganalisis pesan pengguna menggunakan regex untuk mendeteksi perintah OS.
    Jika intent terdeteksi namun eksekusi gagal, error dikemas dalam IntentResult
    agar dapat dikirim ke Ollama sebagai konteks.

    Args:
        user_message: Pesan teks dari pengguna.

    Returns:
        IntentResult jika ada intent OS yang terdeteksi, None jika chat biasa.
    """
    if not user_message or not isinstance(user_message, str):
        logger.warning("Input pesan tidak valid: %r", user_message)
        return None

    message = user_message.strip()
    logger.debug("Memproses intent untuk pesan: '%s'", message)

    # --- PRIORITAS 1: VS Code ---
    if _VSCODE_PATTERN.search(message):
        logger.info("Intent terdeteksi: Buka VS Code")
        success, msg = open_vs_code()
        return _build_result("VS Code", success, msg)

    # --- PRIORITAS 2: Buka Folder ---
    folder_match = _FOLDER_PATTERN.search(message)
    if folder_match:
        raw_folder = folder_match.group("folder_name").strip()
        normalized = _normalize_folder_name(raw_folder)

        if normalized:
            logger.info("Intent terdeteksi: Buka folder '%s' -> '%s'", raw_folder, normalized)
            success, msg = open_local_folder(normalized)
            return _build_result(f"Folder '{normalized}'", success, msg)
        else:
            # Intent terdetek tapi folder tidak dikenal — kirim konteks ke Ollama
            error_ctx = (
                f"Pengguna ingin membuka folder bernama '{raw_folder}', "
                f"namun folder tersebut tidak ada dalam daftar yang diizinkan. "
                f"Berikan saran folder yang tersedia atau minta klarifikasi."
            )
            logger.warning("Folder '%s' tidak dikenal, kirim konteks ke Ollama.", raw_folder)
            return IntentResult(
                detected=True,
                success=False,
                message=f"Folder '{raw_folder}' tidak dikenal dalam sistem.",
                error_context=error_ctx,
            )

    # --- PRIORITAS 3: Buka Browser ---
    browser_match = _BROWSER_PATTERN.search(message)
    if browser_match:
        raw_site = browser_match.group("site_name").strip()
        # Filter terlalu pendek atau tidak relevan (mis: "browser" saja tanpa target)
        if len(raw_site) < 2 or raw_site.lower() in ("browser", "web", "website", "situs"):
            # Tidak cukup info, bukan perintah yang valid
            return None

        normalized_site = _normalize_site_name(raw_site)
        if normalized_site:
            logger.info("Intent terdeteksi: Buka browser '%s' -> '%s'", raw_site, normalized_site)
            success, msg = open_browser_tab(normalized_site)
            return _build_result(f"Website '{normalized_site}'", success, msg)
        else:
            error_ctx = (
                f"Pengguna ingin membuka website '{raw_site}' di browser, "
                f"namun site tersebut tidak ada dalam daftar yang diizinkan. "
                f"Berikan alternatif atau minta klarifikasi URL yang dimaksud."
            )
            return IntentResult(
                detected=True,
                success=False,
                message=f"Website '{raw_site}' tidak ada dalam daftar yang diizinkan.",
                error_context=error_ctx,
            )

    # Tidak ada intent OS yang terdeteksi
    return None


def _build_result(target_name: str, success: bool, message: str) -> IntentResult:
    """Helper membangun IntentResult dari hasil eksekusi OS."""
    error_context = ""
    if not success:
        error_context = (
            f"Pengguna mencoba membuka {target_name} namun gagal dengan pesan: '{message}'. "
            f"Berikan solusi atau langkah troubleshooting yang ramah pengguna dalam bahasa Indonesia."
        )
        logger.warning("Eksekusi OS gagal untuk '%s': %s", target_name, message)

    return IntentResult(
        detected=True,
        success=success,
        message=message,
        error_context=error_context,
    )