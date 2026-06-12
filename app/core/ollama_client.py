import requests
from app.core.config import settings

class OllamaClient:
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.OLLAMA_MODEL

    def generate_response(self, prompt: str, system_prompt: str = None) -> str:
        """
        Mengirim prompt ke Ollama lokal dan mengembalikan jawaban dalam bentuk teks.
        """
        url = f"{self.base_url}/api/generate"
        
        # Menyusun payload data yang dikirim ke Ollama
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False # Set False agar jawaban dikembalikan utuh (bukan per kata)
        }
        
        if system_prompt:
            payload["system"] = system_prompt

        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json().get("response", "Maaf, tidak ada respon dari model.")
        except requests.exceptions.RequestException as e:
            return f"Gagal terhubung ke Ollama. Pastikan Ollama sudah dijalankan! Error: {e}"

# Inisialisasi client agar bisa langsung di-import oleh module lain
ollama_client = OllamaClient()