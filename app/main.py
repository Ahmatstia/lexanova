"""
main.py - Backend Utama Lexa Assistant
Mengintegrasikan OS-Commander, Study Buddy (RAG), dan Workspace Monitor
ke dalam satu FastAPI application.
"""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import shutil
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.ollama_client import ollama_client
from app.modules.commander.intent_parser import parse_and_execute_intent
from app.modules.study_buddy.vector_store import build_rag_context, get_collection_stats
from app.modules.monitor.vision_handler import vision_handler
from app.modules.monitor.ann_model import ann_model_manager

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System Prompt Utama Lexa AI
# ---------------------------------------------------------------------------

LEXA_SYSTEM_PROMPT = (
    "Kamu adalah Lexa AI, asisten pribadi lokal berbasis Llama 3 yang berjalan "
    "100% offline di laptop milik Lexa. "
    "Spesifikasi laptop: CPU AMD Ryzen 5 6600H, RAM 24GB, GPU NVIDIA RTX 3050 (6GB VRAM). "
    "Selalu jawab dalam Bahasa Indonesia yang ringkas, solutif, cerdas, dan personal. "
    "Dilarang keras mengaku sebagai sistem cloud, ChatGPT, atau model AI eksternal lainnya. "
    "Kamu adalah asisten lokal yang berjalan sepenuhnya di mesin Lexa."
)

# ---------------------------------------------------------------------------
# Lifespan Event: Startup & Shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Mengelola event startup dan shutdown aplikasi."""
    logger.info("=" * 60)
    logger.info("🚀 Lexa Assistant Backend Memulai...")
    logger.info("   Ollama URL  : %s", settings.OLLAMA_BASE_URL)
    logger.info("   Model       : %s", settings.OLLAMA_MODEL)
    logger.info("   Project Root: %s", settings.PROJECT_ROOT)
    logger.info("=" * 60)

    # Pre-load ANN model info saat startup
    try:
        model_info = ann_model_manager.get_model_info()
        logger.info(
            "ANN Model siap: %d parameter, device: %s",
            model_info["total_parameters"],
            model_info["device"],
        )
    except Exception as e:
        logger.warning("Gagal memuat info ANN model saat startup: %s", e)

    yield  # Aplikasi berjalan di sini

    # Shutdown: pastikan kamera dirilis
    logger.info("🛑 Lexa Assistant Backend Berhenti. Membersihkan resource...")
    if vision_handler.get_status()["is_active"]:
        vision_handler.stop_camera()
        logger.info("Kamera dirilis saat shutdown.")
    logger.info("Goodbye! 👋")


# ---------------------------------------------------------------------------
# FastAPI Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Lexa Assistant API",
    description=(
        "Backend Unified AI Agent untuk Lexa Assistant. "
        "Menggabungkan OS-Commander, Study Buddy (RAG), dan Workspace Monitor."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# CORS (untuk frontend lokal)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request/Response Models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Request model untuk endpoint /api/chat."""
    pesan: str = Field(..., min_length=1, max_length=4096, description="Pesan dari pengguna")
    use_rag: bool = Field(default=True, description="Aktifkan RAG (Study Buddy) jika True")

    class Config:
        json_schema_extra = {
            "example": {
                "pesan": "Jelaskan tentang metodologi penelitian dalam skripsi saya",
                "use_rag": True,
            }
        }


class ChatResponse(BaseModel):
    """Response model untuk endpoint /api/chat."""
    user_chat: str
    lexa_response: str
    source: str  # "os_commander" | "rag" | "ollama_direct"
    processing_time_ms: float
    rag_context_used: bool = False
    os_action_success: Optional[bool] = None


class MonitorStatusResponse(BaseModel):
    """Response model untuk status monitor."""
    is_active: bool
    camera_index: int
    frame_count: int
    face_detected: bool
    error_message: str
    latest_detection: Optional[dict]
    features: dict


class IngestResponse(BaseModel):
    """Response model untuk ingest dokumen."""
    success: bool
    message: str
    documents_found: int = 0


# ---------------------------------------------------------------------------
# Endpoint: Root
# ---------------------------------------------------------------------------

@app.get("/", tags=["System"])
def read_root():
    """Health check endpoint."""
    return {
        "status": "online",
        "app": "Lexa Assistant Backend",
        "version": "2.0.0",
        "message": "Lexa Assistant berjalan 100% Offline! 🚀",
    }


# ---------------------------------------------------------------------------
# ENDPOINT UTAMA: /api/chat
# ---------------------------------------------------------------------------

@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
def chat_with_lexa(request: ChatRequest):
    """
    Endpoint chat utama dengan pipeline tiga tahap:
    1. OS-Commander: deteksi perintah sistem
    2. Study Buddy (RAG): cari konteks dari dokumen lokal
    3. Ollama Direct: chat standar ke Llama 3
    """
    start_time = time.time()
    pesan = request.pesan.strip()

    logger.info("Chat request diterima: '%s'", pesan[:80])

    # =========================================================
    # TAHAP 1: OS-Commander — Deteksi Intent Sistem
    # =========================================================
    intent_result = parse_and_execute_intent(pesan)

    if intent_result is not None and intent_result.detected:
        if intent_result.success:
            # Eksekusi OS berhasil, kembalikan langsung
            elapsed_ms = (time.time() - start_time) * 1000
            logger.info("OS-Commander berhasil: %s", intent_result.message)
            return ChatResponse(
                user_chat=pesan,
                lexa_response=f"✅ {intent_result.message}",
                source="os_commander",
                processing_time_ms=round(elapsed_ms, 2),
                os_action_success=True,
            )
        else:
            # Eksekusi OS gagal — kirim error context ke Ollama untuk solusi
            logger.warning("OS-Commander gagal: %s", intent_result.message)
            error_aware_prompt = (
                f"{intent_result.error_context}\n\n"
                f"Pesan asli pengguna: {pesan}"
            )
            ollama_response = ollama_client.generate_response(
                prompt=error_aware_prompt,
                system_prompt=LEXA_SYSTEM_PROMPT,
            )
            elapsed_ms = (time.time() - start_time) * 1000
            return ChatResponse(
                user_chat=pesan,
                lexa_response=f"⚠️ {intent_result.message}\n\n💡 {ollama_response}",
                source="os_commander",
                processing_time_ms=round(elapsed_ms, 2),
                os_action_success=False,
            )

    # =========================================================
    # TAHAP 2: Study Buddy — RAG dari Dokumen Lokal
    # =========================================================
    rag_context = None
    if request.use_rag:
        try:
            rag_context = build_rag_context(query=pesan, k=3)
        except Exception as e:
            logger.warning("RAG gagal, lanjut ke Ollama direct: %s", e)
            rag_context = None

    if rag_context:
        # Ada konteks relevan dari dokumen
        rag_system_prompt = (
            f"{LEXA_SYSTEM_PROMPT}\n\n"
            f"Berikut adalah informasi relevan dari dokumen lokal pengguna yang dapat "
            f"kamu gunakan sebagai referensi untuk menjawab pertanyaan:\n\n"
            f"=== KONTEKS DOKUMEN ===\n{rag_context}\n=== END KONTEKS ===\n\n"
            f"Gunakan informasi di atas jika relevan. Jika pertanyaan tidak berkaitan "
            f"dengan konteks, jawab berdasarkan pengetahuanmu sendiri."
        )
        ollama_response = ollama_client.generate_response(
            prompt=pesan,
            system_prompt=rag_system_prompt,
        )
        elapsed_ms = (time.time() - start_time) * 1000
        logger.info("Respons dari RAG (Study Buddy). Durasi: %.2f ms", elapsed_ms)
        return ChatResponse(
            user_chat=pesan,
            lexa_response=ollama_response,
            source="rag",
            processing_time_ms=round(elapsed_ms, 2),
            rag_context_used=True,
        )

    # =========================================================
    # TAHAP 3: Ollama Direct — Chat Standar
    # =========================================================
    ollama_response = ollama_client.generate_response(
        prompt=pesan,
        system_prompt=LEXA_SYSTEM_PROMPT,
    )
    elapsed_ms = (time.time() - start_time) * 1000
    logger.info("Respons dari Ollama direct. Durasi: %.2f ms", elapsed_ms)
    return ChatResponse(
        user_chat=pesan,
        lexa_response=ollama_response,
        source="ollama_direct",
        processing_time_ms=round(elapsed_ms, 2),
    )


# ---------------------------------------------------------------------------
# ENDPOINT: /api/chat (GET - backward compatibility)
# ---------------------------------------------------------------------------

@app.get("/api/chat", tags=["Chat"])
def chat_with_lexa_get(
    pesan: str = Query(..., min_length=1, max_length=4096, description="Pesan dari pengguna"),
    use_rag: bool = Query(default=True),
):
    """GET versi /api/chat untuk kemudahan testing via browser."""
    return chat_with_lexa(ChatRequest(pesan=pesan, use_rag=use_rag))


# ---------------------------------------------------------------------------
# ENDPOINT: Study Buddy — Ingest Dokumen
# ---------------------------------------------------------------------------

@app.post("/api/study/ingest", response_model=IngestResponse, tags=["Study Buddy"])
def ingest_documents(background_tasks: BackgroundTasks):
    """
    Memuat semua dokumen dari data/documents/ dan mengindeksnya ke ChromaDB.
    Operasi berjalan di background untuk tidak memblokir server.
    """
    def _run_ingest():
        from app.modules.study_buddy.document_loader import load_and_split_documents
        from app.modules.study_buddy.vector_store import ingest_documents as vs_ingest

        logger.info("Memulai ingest dokumen di background...")
        chunks = load_and_split_documents()
        if not chunks:
            logger.warning("Tidak ada chunk untuk diindeks.")
            return
        success, msg = vs_ingest(chunks)
        logger.info("Hasil ingest: %s — %s", success, msg)

    background_tasks.add_task(_run_ingest)
    return IngestResponse(
        success=True,
        message="Proses ingest dokumen dimulai di background. Cek log untuk status.",
    )


@app.post("/api/study/upload", tags=["Study Buddy"])
def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Mengunggah dokumen PDF/TXT/MD ke folder data/documents/ 
    dan otomatis memicu re-ingest di background.
    """
    allowed_extensions = {".pdf", ".txt", ".md"}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Ekstensi file tidak didukung. Harap upload: {', '.join(allowed_extensions)}"
        )
    
    # Simpan file ke direktori dokumen
    file_path = settings.DOCUMENTS_DIR / file.filename
    try:
        settings.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"File berhasil diunggah: {file_path}")
    except Exception as e:
        logger.error(f"Gagal menyimpan file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Gagal menyimpan file: {e}")
    
    # Jalankan ingest di background sama seperti endpoint /api/study/ingest
    def _run_ingest():
        from app.modules.study_buddy.document_loader import load_and_split_documents
        from app.modules.study_buddy.vector_store import ingest_documents as vs_ingest
        logger.info(f"Memulai ingest otomatis setelah upload file {file.filename}...")
        chunks = load_and_split_documents()
        if chunks:
            vs_ingest(chunks)
            
    background_tasks.add_task(_run_ingest)
    
    return {
        "success": True,
        "message": f"File '{file.filename}' berhasil diunggah dan sedang di-indeks.",
        "filename": file.filename
    }


@app.get("/api/study/status", tags=["Study Buddy"])
def get_study_status():
    """Mengembalikan statistik koleksi dokumen di ChromaDB."""
    try:
        stats = get_collection_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal mengambil statistik: {e}") from e


@app.get("/api/study/documents", tags=["Study Buddy"])
def list_documents():
    """Menampilkan daftar dokumen yang tersedia di folder data/documents/."""
    from app.modules.study_buddy.document_loader import list_available_documents
    try:
        docs = list_available_documents()
        return {
            "total": len(docs),
            "documents": docs,
            "instructions": (
                "Tambahkan file PDF/TXT/MD ke folder 'data/documents/' "
                "lalu panggil POST /api/study/ingest untuk mengindeks."
            ),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# ---------------------------------------------------------------------------
# ENDPOINT: Workspace Monitor
# ---------------------------------------------------------------------------

@app.post("/api/monitor/start", tags=["Workspace Monitor"])
def start_monitor(camera_index: int = Query(default=0, ge=0, le=9)):
    """
    Mengaktifkan kamera dan memulai deteksi workspace (On-Demand Activation).
    Kamera hanya aktif saat endpoint ini dipanggil untuk hemat VRAM.
    """
    success, msg = vision_handler.start_camera(camera_index=camera_index)
    if not success:
        raise HTTPException(status_code=503, detail=msg)
    return {"success": True, "message": msg}


@app.post("/api/monitor/stop", tags=["Workspace Monitor"])
def stop_monitor():
    """
    Menonaktifkan kamera dan merilis semua resource (cap.release()).
    Penting untuk dipanggil saat monitor tidak diperlukan.
    """
    success, msg = vision_handler.stop_camera()
    if not success:
        raise HTTPException(status_code=500, detail=msg)
    return {"success": True, "message": msg}


@app.get("/api/monitor/status", response_model=MonitorStatusResponse, tags=["Workspace Monitor"])
def get_monitor_status():
    """Mengembalikan status lengkap workspace monitor dan deteksi terbaru."""
    status = vision_handler.get_status()
    return MonitorStatusResponse(**status)


@app.get("/api/monitor/analyze", tags=["Workspace Monitor"])
def analyze_posture():
    """
    Mengambil snapshot dari kamera aktif dan menjalankan analisis postur menggunakan ANN.
    Kamera harus diaktifkan terlebih dahulu via /api/monitor/start.
    """
    monitor_status = vision_handler.get_status()

    if not monitor_status["is_active"]:
        raise HTTPException(
            status_code=400,
            detail="Kamera tidak aktif. Aktifkan terlebih dahulu via POST /api/monitor/start",
        )

    try:
        # Ambil data dari deteksi terbaru
        latest = monitor_status.get("latest_detection") or {}
        features = monitor_status.get("features") or {}

        # Jalankan inferensi ANN
        result = ann_model_manager.infer(
            face_locations=[],  # Akan diisi dari vision handler
            brightness=latest.get("brightness", 128.0),
        )

        # Jika kamera aktif dan ada deteksi, gunakan data aktual
        snapshot = vision_handler.capture_snapshot()
        if snapshot is not None:
            import numpy as np
            brightness = float(np.mean(snapshot))

            # Re-run deteksi pada snapshot untuk data akurat
            det_result = vision_handler._process_frame(snapshot)
            result = ann_model_manager.infer(
                face_locations=det_result.face_locations,
                frame_width=snapshot.shape[1],
                frame_height=snapshot.shape[0],
                brightness=det_result.brightness,
            )

        return {
            "status": "analyzed",
            "prediction": {
                "class": result.predicted_class,
                "label": result.predicted_label,
                "confidence": round(result.confidence, 4),
                "is_anomaly": result.is_anomaly,
                "recommendation": result.recommendation,
                "all_probabilities": result.all_probabilities,
            },
        }

    except Exception as e:
        logger.error("Error saat analisis postur: %s", e)
        raise HTTPException(status_code=500, detail=f"Analisis gagal: {e}") from e


@app.get("/api/monitor/model-info", tags=["Workspace Monitor"])
def get_ann_model_info():
    """Menampilkan informasi arsitektur model ANN."""
    try:
        return ann_model_manager.get_model_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


# ---------------------------------------------------------------------------
# Jalankan server (untuk development langsung)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )