"""
vision_handler.py - Modul Workspace Monitor untuk Lexa Assistant
Mengelola akses webcam dengan pola On-Demand Activation untuk efisiensi VRAM/RAM.
Menggunakan OpenCV dan Haar Cascade untuk deteksi wajah berbasis DIP.
"""

import logging
import threading
import time
import gc
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# --- Konfigurasi ---
DEFAULT_CAMERA_INDEX = 0        # Index webcam (0 = kamera default)
FRAME_WIDTH = 640               # Resolusi rendah untuk efisiensi
FRAME_HEIGHT = 480
FACE_SCALE_FACTOR = 1.1         # Parameter Haar Cascade
FACE_MIN_NEIGHBORS = 5
FACE_MIN_SIZE = (80, 80)        # Ukuran wajah minimum yang dideteksi
PRESENCE_COOLDOWN_SEC = 3.0     # Interval minimum antar-deteksi untuk hindari spam log


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class MonitorStatus:
    """Status aktif dari workspace monitor."""
    is_active: bool = False
    camera_index: int = DEFAULT_CAMERA_INDEX
    frame_count: int = 0
    face_detected: bool = False
    last_detection_time: float = 0.0
    error_message: str = ""
    features: dict = field(default_factory=dict)


@dataclass
class DetectionResult:
    """Hasil dari satu frame deteksi."""
    face_detected: bool
    face_count: int
    face_locations: list  # List of (x, y, w, h)
    brightness: float     # Rata-rata brightness frame (0-255)
    timestamp: float


# ---------------------------------------------------------------------------
# Vision Handler Class
# ---------------------------------------------------------------------------

class VisionHandler:
    """
    Mengelola siklus hidup kamera (On-Demand Activation).
    Thread-safe: bisa dipanggil dari multiple request FastAPI.
    """

    def __init__(self) -> None:
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._status = MonitorStatus()
        self._face_cascade: Optional[cv2.CascadeClassifier] = None
        self._bg_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._latest_result: Optional[DetectionResult] = None

        # Muat Haar Cascade (built-in OpenCV, tidak perlu file eksternal)
        self._load_face_cascade()

    def _load_face_cascade(self) -> None:
        """Memuat Haar Cascade untuk deteksi wajah dari OpenCV built-in data."""
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._face_cascade = cv2.CascadeClassifier(cascade_path)

            if self._face_cascade.empty():
                logger.error("Gagal memuat Haar Cascade. File tidak valid: %s", cascade_path)
                self._face_cascade = None
            else:
                logger.info("Haar Cascade berhasil dimuat dari: %s", cascade_path)
        except Exception as e:
            logger.error("Error saat memuat Haar Cascade: %s", e)
            self._face_cascade = None

    # -----------------------------------------------------------------------
    # Kontrol Kamera
    # -----------------------------------------------------------------------

    def start_camera(self, camera_index: int = DEFAULT_CAMERA_INDEX) -> Tuple[bool, str]:
        """
        Mengaktifkan kamera dan memulai thread deteksi background.
        Jika kamera sudah aktif, mengembalikan status OK.

        Args:
            camera_index: Index kamera (default: 0).

        Returns:
            Tuple (status_sukses: bool, pesan: str)
        """
        with self._lock:
            if self._status.is_active and self._cap is not None:
                return (True, f"Kamera (index: {camera_index}) sudah aktif.")

            logger.info("Mengaktifkan kamera index: %d", camera_index)
            try:
                cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)  # CAP_DSHOW lebih stabil di Windows

                if not cap.isOpened():
                    # Fallback tanpa CAP_DSHOW
                    cap = cv2.VideoCapture(camera_index)

                if not cap.isOpened():
                    msg = f"Tidak dapat membuka kamera index {camera_index}. Periksa koneksi webcam."
                    logger.error(msg)
                    self._status.error_message = msg
                    return (False, msg)

                # Set resolusi rendah untuk efisiensi
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
                cap.set(cv2.CAP_PROP_FPS, 15)  # Frame rate rendah untuk hemat resource

                self._cap = cap
                self._status.is_active = True
                self._status.camera_index = camera_index
                self._status.frame_count = 0
                self._status.error_message = ""
                self._stop_event.clear()

                # Mulai thread deteksi background
                self._bg_thread = threading.Thread(
                    target=self._detection_loop,
                    daemon=True,
                    name="VisionDetectionThread",
                )
                self._bg_thread.start()

                logger.info("Kamera index %d berhasil diaktifkan.", camera_index)
                return (True, f"Kamera (index: {camera_index}) berhasil diaktifkan.")

            except Exception as e:
                msg = f"Error saat mengaktifkan kamera: {type(e).__name__}: {e}"
                logger.exception(msg)
                self._release_camera_internal()
                return (False, msg)

    def stop_camera(self) -> Tuple[bool, str]:
        """
        Menonaktifkan kamera dan merilis semua resource (cap.release()).
        Kritis untuk efisiensi VRAM/RAM: selalu panggil ini jika kamera tidak diperlukan.

        Returns:
            Tuple (status_sukses: bool, pesan: str)
        """
        with self._lock:
            if not self._status.is_active:
                return (True, "Kamera sudah dalam kondisi nonaktif.")

            logger.info("Menonaktifkan kamera...")
            self._stop_event.set()

        # Tunggu thread berhenti (tanpa lock untuk hindari deadlock)
        if self._bg_thread and self._bg_thread.is_alive():
            self._bg_thread.join(timeout=3.0)
            if self._bg_thread.is_alive():
                logger.warning("Thread deteksi tidak berhenti dalam 3 detik.")

        with self._lock:
            self._release_camera_internal()
            total_frames = self._status.frame_count
            self._status.is_active = False
            self._status.face_detected = False
            self._latest_result = None

        gc.collect()  # Bersihkan memori setelah rilis
        msg = f"Kamera berhasil dinonaktifkan. Total frame diproses: {total_frames}."
        logger.info(msg)
        return (True, msg)

    def _release_camera_internal(self) -> None:
        """Internal: melepas resource kamera (panggil dalam lock)."""
        if self._cap is not None:
            try:
                self._cap.release()
                logger.debug("cap.release() dipanggil.")
            except Exception as e:
                logger.warning("Error saat release kamera: %s", e)
            finally:
                self._cap = None

    # -----------------------------------------------------------------------
    # Detection Loop (Background Thread)
    # -----------------------------------------------------------------------

    def _detection_loop(self) -> None:
        """
        Loop deteksi yang berjalan di background thread.
        Baca frame -> proses DIP -> deteksi wajah -> update status.
        """
        logger.info("Thread deteksi dimulai.")
        last_log_time = 0.0

        while not self._stop_event.is_set():
            cap = self._cap
            if cap is None or not cap.isOpened():
                logger.warning("Kamera tidak tersedia di detection loop. Berhenti.")
                break

            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning("Gagal membaca frame dari kamera.")
                time.sleep(0.1)
                continue

            with self._lock:
                self._status.frame_count += 1

            # Proses DIP dan deteksi
            result = self._process_frame(frame)

            with self._lock:
                self._latest_result = result
                self._status.face_detected = result.face_detected
                self._status.features = {
                    "face_count": result.face_count,
                    "brightness": round(result.brightness, 2),
                }

                # Log deteksi (cooldown untuk hindari spam)
                now = time.time()
                if result.face_detected and (now - last_log_time) > PRESENCE_COOLDOWN_SEC:
                    logger.info(
                        "Wajah terdeteksi: %d wajah, brightness: %.2f",
                        result.face_count, result.brightness,
                    )
                    last_log_time = now

            # Frame rate ~10 FPS (100ms sleep) untuk hemat CPU/GPU
            time.sleep(0.1)

        logger.info("Thread deteksi berhenti.")

    # -----------------------------------------------------------------------
    # Digital Image Processing (DIP)
    # -----------------------------------------------------------------------

    def _process_frame(self, frame: np.ndarray) -> DetectionResult:
        """
        Memproses satu frame dengan teknik DIP:
        1. Grayscale conversion (efisiensi)
        2. Histogram equalization (normalisasi pencahayaan)
        3. Gaussian blur (reduksi noise)
        4. Haar Cascade face detection

        Args:
            frame: Frame BGR dari VideoCapture.

        Returns:
            DetectionResult berisi informasi deteksi.
        """
        face_locations = []
        face_detected = False
        brightness = 0.0

        try:
            # Hitung brightness dari frame asli
            brightness = float(np.mean(frame))

            if self._face_cascade is None:
                return DetectionResult(
                    face_detected=False,
                    face_count=0,
                    face_locations=[],
                    brightness=brightness,
                    timestamp=time.time(),
                )

            # === DIP Pipeline ===
            # 1. Convert ke Grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 2. Histogram Equalization (normalisasi pencahayaan)
            gray_eq = cv2.equalizeHist(gray)

            # 3. Gaussian Blur (reduksi noise sebelum deteksi)
            gray_blur = cv2.GaussianBlur(gray_eq, (5, 5), 0)

            # 4. Deteksi Wajah dengan Haar Cascade
            faces = self._face_cascade.detectMultiScale(
                gray_blur,
                scaleFactor=FACE_SCALE_FACTOR,
                minNeighbors=FACE_MIN_NEIGHBORS,
                minSize=FACE_MIN_SIZE,
                flags=cv2.CASCADE_SCALE_IMAGE,
            )

            if len(faces) > 0:
                face_locations = faces.tolist()
                face_detected = True

        except cv2.error as e:
            logger.error("OpenCV error saat proses frame: %s", e)
        except Exception as e:
            logger.error("Error tak terduga saat proses frame: %s — %s", type(e).__name__, e)

        return DetectionResult(
            face_detected=face_detected,
            face_count=len(face_locations),
            face_locations=face_locations,
            brightness=brightness,
            timestamp=time.time(),
        )

    # -----------------------------------------------------------------------
    # Status & Snapshot
    # -----------------------------------------------------------------------

    def get_status(self) -> dict:
        """
        Mengembalikan status lengkap workspace monitor.

        Returns:
            Dict berisi status kamera dan hasil deteksi terbaru.
        """
        with self._lock:
            latest = self._latest_result
            return {
                "is_active": self._status.is_active,
                "camera_index": self._status.camera_index,
                "frame_count": self._status.frame_count,
                "face_detected": self._status.face_detected,
                "error_message": self._status.error_message,
                "latest_detection": {
                    "face_count": latest.face_count if latest else 0,
                    "brightness": latest.brightness if latest else 0.0,
                    "timestamp": latest.timestamp if latest else 0.0,
                } if latest else None,
                "features": self._status.features,
            }

    def capture_snapshot(self) -> Optional[np.ndarray]:
        """
        Mengambil satu frame dari kamera (jika aktif).

        Returns:
            Frame numpy array atau None jika kamera tidak aktif.
        """
        with self._lock:
            if not self._status.is_active or self._cap is None:
                logger.warning("Snapshot gagal: kamera tidak aktif.")
                return None

        cap = self._cap
        if cap and cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                return frame
        return None


# ---------------------------------------------------------------------------
# Singleton Instance
# ---------------------------------------------------------------------------

vision_handler = VisionHandler()
