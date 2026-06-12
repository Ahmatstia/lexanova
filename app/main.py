from fastapi import FastAPI
from app.core.ollama_client import ollama_client
from app.modules.commander.intent_parser import parse_and_execute_intent

app = FastAPI(title="Lexa Assistant API", version="1.0.0")

@app.get("/")
def read_root():
    return {"message": "Lexa Assistant Backend is Running Offline!"}

@app.get("/api/chat")
def chat_with_lexa(pesan: str):
    """
    Endpoint utama yang mendeteksi perintah sistem (OS-Commander) 
    sebelum melemparkannya ke Ollama.
    """
    # LANGKAH 1: Cek apakah chat mengandung perintah sistem lokal (NLP Intent Parsing)
    system_action_response = parse_and_execute_intent(pesan)
    
    # Jika terdeteksi perintah sistem, langsung kembalikan hasilnya tanpa lewat Ollama
    if system_action_response:
        return {
            "user_chat": pesan,
            "lexa_response": f"[OS-Commander]: Perintah dieksekusi! {system_action_response}"
        }
    
    # LANGKAH 2: Jika hanya chat biasa, lempar ke Ollama Llama 3
    system_instruction = (
        "Kamu adalah Lexa AI, asisten pribadi berbasis lokal yang super pintar. "
        "Kamu berjalan 100% OFFLINE secara lokal di laptop milik user yang bernama 'Lexa'. "
        "Spesifikasi laptop ini: CPU AMD Ryzen 5 6600H, RAM 24 GB, GPU RTX 3050 (6GB VRAM). "
        "Selalu jawab dalam bahasa Indonesia yang ringkas dan ramah."
    )
    
    respons_ai = ollama_client.generate_response(prompt=pesan, system_prompt=system_instruction)
    
    return {
        "user_chat": pesan,
        "lexa_response": respons_ai
    }