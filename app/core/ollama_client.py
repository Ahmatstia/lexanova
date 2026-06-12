"""
ollama_client.py - Client Ollama untuk Lexa Assistant
Mengirim prompt ke Ollama lokal dengan timeout, retry, dan error handling lengkap.
"""

import logging
import time
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from app.core.config import settings

logger = logging.getLogger(__name__)

# Konfigurasi retry strategy
_RETRY_STRATEGY = Retry(
    total=2,                    # Maksimum 2 retry
    backoff_factor=0.5,         # Jeda: 0.5s, 1s
    status_forcelist=[500, 502, 503, 504],
    allowed_methods=["POST"],
)


class OllamaClient:
    """
    Client untuk berkomunikasi dengan Ollama API secara lokal.
    Mendukung timeout yang dikonfigurasi, retry otomatis, dan error handling robust.
    """

    def __init__(self) -> None:
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.OLLAMA_MODEL
        self.timeout = settings.OLLAMA_TIMEOUT_SEC

        # Session dengan retry otomatis
        self._session = requests.Session()
        adapter = HTTPAdapter(max_retries=_RETRY_STRATEGY)
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

        logger.info(
            "OllamaClient diinisialisasi: url=%s, model=%s, timeout=%ds",
            self.base_url, self.model, self.timeout,
        )

    def is_available(self) -> bool:
        """
        Mengecek apakah Ollama server dapat dijangkau.

        Returns:
            True jika server online dan merespons.
        """
        try:
            resp = self._session.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def generate_response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Mengirim prompt ke Ollama dan mengembalikan respons teks.

        Args:
            prompt: Pesan/pertanyaan pengguna.
            system_prompt: Instruksi sistem untuk mendefinisikan perilaku AI.
            temperature: Kreativitas respons (0.0 = deterministik, 1.0 = kreatif).
            max_tokens: Batas panjang respons (None = default model).

        Returns:
            String respons dari Ollama, atau pesan error yang informatif.
        """
        if not prompt or not isinstance(prompt, str):
            return "Input prompt tidak valid."

        if len(prompt.strip()) == 0:
            return "Pesan tidak boleh kosong."

        url = f"{self.base_url}/api/generate"
        payload: dict = {
            "model": self.model,
            "prompt": prompt.strip(),
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens or -1,  # -1 = unlimited
            },
        }

        if system_prompt:
            payload["system"] = system_prompt

        start_time = time.time()
        logger.debug("Mengirim request ke Ollama. Prompt length: %d karakter.", len(prompt))

        try:
            response = self._session.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()

            data = response.json()
            respons_text = data.get("response", "").strip()
            elapsed = time.time() - start_time

            if not respons_text:
                logger.warning("Ollama mengembalikan respons kosong.")
                return "Maaf, Ollama tidak menghasilkan respons. Coba lagi."

            logger.info(
                "Ollama merespons dalam %.2fs. Panjang respons: %d karakter.",
                elapsed, len(respons_text),
            )
            return respons_text

        except requests.exceptions.ConnectionError:
            msg = (
                f"❌ Tidak dapat terhubung ke Ollama di '{self.base_url}'. "
                f"Pastikan Ollama sudah dijalankan dengan perintah: `ollama serve`"
            )
            logger.error(msg)
            return msg

        except requests.exceptions.Timeout:
            msg = (
                f"⏱️ Request ke Ollama melebihi batas waktu ({self.timeout} detik). "
                f"Model mungkin sedang dimuat atau terlalu sibuk. Coba lagi sebentar."
            )
            logger.error(msg)
            return msg

        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                msg = (
                    f"❌ Model '{self.model}' tidak ditemukan di Ollama. "
                    f"Unduh dengan: `ollama pull {self.model}`"
                )
            else:
                msg = f"❌ Error HTTP dari Ollama: {e}"
            logger.error(msg)
            return msg

        except ValueError as e:
            msg = f"❌ Gagal memparse respons JSON dari Ollama: {e}"
            logger.error(msg)
            return msg

        except Exception as e:
            msg = f"❌ Error tak terduga saat berkomunikasi dengan Ollama: {type(e).__name__}: {e}"
            logger.exception(msg)
            return msg

    def list_available_models(self) -> list:
        """
        Mendapatkan daftar model yang tersedia di Ollama lokal.

        Returns:
            List nama model, atau list kosong jika gagal.
        """
        try:
            resp = self._session.get(f"{self.base_url}/api/tags", timeout=10)
            resp.raise_for_status()
            models = resp.json().get("models", [])
            return [m.get("name", "") for m in models]
        except Exception as e:
            logger.warning("Gagal mendapatkan daftar model dari Ollama: %s", e)
            return []


# Singleton client
ollama_client = OllamaClient()