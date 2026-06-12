"""
os_actions.py - Modul OS-Commander untuk Lexa Assistant
Menyediakan fungsi kontrol sistem operasi yang aman dan tervalidasi.
"""

import os
import re
import subprocess
import webbrowser
import logging
from pathlib import Path
from typing import Tuple

# Konfigurasi logger untuk modul ini
logger = logging.getLogger(__name__)

# --- Root project dideteksi secara dinamis ---
# Asumsi: os_actions.py berada di app/modules/commander/
_MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _MODULE_DIR.parent.parent.parent  # Naik 3 level ke root proyek

# Karakter yang diizinkan untuk nama folder/site (whitelist anti-injection)
_SAFE_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_\- ]+$")

# Daftar folder yang diizinkan (path dibangun secara dinamis dari PROJECT_ROOT)
_ALLOWED_FOLDERS: dict[str, Path] = {
    "skripsi": PROJECT_ROOT / "data" / "documents",
    "download": Path(os.environ.get("USERPROFILE", Path.home())) / "Downloads",
    "project": PROJECT_ROOT,
    "data": PROJECT_ROOT / "data",
}

# Daftar URL yang diizinkan (whitelist, bukan input bebas)
_ALLOWED_URLS: dict[str, str] = {
    "github": "https://github.com",
    "chatgpt": "https://chat.openai.com",
    "gemini": "https://gemini.google.com",
    "youtube": "https://youtube.com",
    "google": "https://google.com",
}


def _validate_safe_name(name: str) -> bool:
    """
    Memvalidasi bahwa nama hanya mengandung karakter yang aman.
    Mencegah path traversal dan command injection.

    Args:
        name: String yang akan divalidasi.

    Returns:
        True jika nama aman, False jika mengandung karakter berbahaya.
    """
    if not name or not isinstance(name, str):
        return False
    if len(name) > 128:  # Batasi panjang input
        return False
    return bool(_SAFE_NAME_PATTERN.match(name.strip()))


def open_vs_code(target_path: str = None) -> Tuple[bool, str]:
    """
    Membuka VS Code. Jika target_path diberikan, membuka folder/file tersebut.
    Path divalidasi untuk mencegah command injection.

    Args:
        target_path: (Opsional) Path absolut folder atau file yang akan dibuka.

    Returns:
        Tuple (status_sukses: bool, pesan: str)
    """
    try:
        cmd = ["code"]

        if target_path:
            # Validasi: path harus berada di dalam PROJECT_ROOT atau lokasi yang diizinkan
            resolved = Path(target_path).resolve()

            # Cegah path traversal ke luar project
            allowed_roots = [PROJECT_ROOT, Path(os.environ.get("USERPROFILE", Path.home()))]
            is_safe = any(str(resolved).startswith(str(root)) for root in allowed_roots)

            if not is_safe:
                logger.warning("Percobaan akses path tidak aman: %s", target_path)
                return (False, f"Akses ke path '{target_path}' ditolak karena alasan keamanan.")

            if not resolved.exists():
                return (False, f"Path '{resolved}' tidak ditemukan di sistem.")

            cmd.append(str(resolved))

        # Jalankan tanpa shell=True untuk mencegah shell injection
        subprocess.Popen(cmd, shell=False)
        target_info = f" di '{target_path}'" if target_path else ""
        logger.info("VS Code berhasil dibuka%s.", target_info)
        return (True, f"VS Code berhasil dibuka{target_info}!")

    except FileNotFoundError:
        msg = (
            "Perintah 'code' tidak ditemukan. "
            "Pastikan VS Code sudah terinstall dan terdaftar di PATH sistem."
        )
        logger.error(msg)
        return (False, msg)
    except PermissionError as e:
        msg = f"Tidak ada izin untuk membuka VS Code. Error: {e}"
        logger.error(msg)
        return (False, msg)
    except Exception as e:
        msg = f"Terjadi kesalahan tak terduga saat membuka VS Code. Error: {type(e).__name__}: {e}"
        logger.exception(msg)
        return (False, msg)


def open_local_folder(folder_type: str) -> Tuple[bool, str]:
    """
    Membuka folder lokal berdasarkan alias yang terdaftar di _ALLOWED_FOLDERS.
    Path dibangun secara dinamis, tidak di-hardcode.

    Args:
        folder_type: Alias folder (mis: 'skripsi', 'download', 'project').

    Returns:
        Tuple (status_sukses: bool, pesan: str)
    """
    if not _validate_safe_name(folder_type):
        msg = f"Nama folder '{folder_type}' mengandung karakter yang tidak valid."
        logger.warning(msg)
        return (False, msg)

    folder_key = folder_type.strip().lower()
    target_path = _ALLOWED_FOLDERS.get(folder_key)

    if target_path is None:
        available = ", ".join(_ALLOWED_FOLDERS.keys())
        msg = f"Folder alias '{folder_type}' tidak dikenal. Tersedia: {available}."
        logger.warning(msg)
        return (False, msg)

    if not target_path.exists():
        # Coba buat folder jika belum ada (khusus untuk folder data)
        if folder_key in ("skripsi", "data"):
            try:
                target_path.mkdir(parents=True, exist_ok=True)
                logger.info("Folder '%s' dibuat secara otomatis.", target_path)
            except OSError as e:
                msg = f"Folder '{target_path}' tidak ada dan gagal dibuat. Error: {e}"
                logger.error(msg)
                return (False, msg)
        else:
            msg = f"Folder '{target_path}' tidak ditemukan di sistem."
            logger.warning(msg)
            return (False, msg)

    try:
        # os.startfile aman karena menerima Path object, bukan shell command
        os.startfile(str(target_path))
        logger.info("Folder '%s' (%s) berhasil dibuka.", folder_type, target_path)
        return (True, f"Folder '{folder_type}' berhasil dibuka di '{target_path}'.")
    except OSError as e:
        msg = f"Gagal membuka folder '{target_path}'. Error: {e}"
        logger.error(msg)
        return (False, msg)
    except Exception as e:
        msg = f"Terjadi kesalahan tak terduga. Error: {type(e).__name__}: {e}"
        logger.exception(msg)
        return (False, msg)


def open_browser_tab(site: str) -> Tuple[bool, str]:
    """
    Membuka tab browser untuk website yang terdaftar di whitelist _ALLOWED_URLS.
    Hanya URL yang sudah diketahui yang dapat dibuka untuk mencegah SSRF.

    Args:
        site: Alias website (mis: 'github', 'youtube') atau URL lengkap (https://...).

    Returns:
        Tuple (status_sukses: bool, pesan: str)
    """
    if not site or not isinstance(site, str):
        return (False, "Input site tidak valid.")

    site_key = site.strip().lower()

    # Cari di whitelist alias
    target_url = _ALLOWED_URLS.get(site_key)

    # Jika tidak ada di alias, cek apakah ini URL valid (https only, tanpa query injection)
    if target_url is None:
        if site_key.startswith("https://") and len(site_key) <= 256:
            # Validasi URL dasar: hanya https, tidak ada karakter shell berbahaya
            if re.match(r"^https://[a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+$", site_key):
                target_url = site_key
            else:
                return (False, f"URL '{site}' mengandung karakter yang tidak aman.")
        else:
            available = ", ".join(_ALLOWED_URLS.keys())
            msg = (
                f"Website '{site}' tidak dikenal. "
                f"Gunakan alias yang tersedia: {available}, "
                f"atau URL lengkap diawali 'https://'."
            )
            return (False, msg)

    try:
        webbrowser.open(target_url)
        logger.info("Browser dibuka untuk URL: %s", target_url)
        return (True, f"Browser berhasil dibuka menuju '{target_url}'.")
    except Exception as e:
        msg = f"Gagal membuka browser. Error: {type(e).__name__}: {e}"
        logger.exception(msg)
        return (False, msg)