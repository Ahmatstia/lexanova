"""
ann_model.py - Modul ANN Classifier untuk Lexa Workspace Monitor
Jaringan Saraf Tiruan ringan (PyTorch) untuk klasifikasi postur/kelelahan.
Desain: low-parameter, memory-efficient, CPU/GPU agnostik.
"""

import logging
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)

# --- Konfigurasi ---
_MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _MODULE_DIR.parent.parent.parent
MODEL_SAVE_PATH = PROJECT_ROOT / "data" / "ann_posture_model.pt"

# Pilih device: GPU jika VRAM tersedia, fallback ke CPU
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > 0
    else "cpu"
)

# Dimensi fitur input: [x_center, y_center, face_width, face_height, brightness, aspect_ratio]
INPUT_FEATURES = 6

# Label kelas output
CLASS_LABELS: Dict[int, str] = {
    0: "Tidak Ada Wajah",      # No face detected
    1: "Postur Normal",         # Normal, attentive posture
    2: "Postur Lelah",          # Fatigue/slouching detected
    3: "Postur Terlalu Dekat",  # Too close to screen
    4: "Postur Terlalu Jauh",   # Too far from screen
}
NUM_CLASSES = len(CLASS_LABELS)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class InferenceResult:
    """Hasil inferensi dari model ANN."""
    predicted_class: int
    predicted_label: str
    confidence: float           # Probabilitas tertinggi (0.0 - 1.0)
    all_probabilities: Dict[str, float]
    is_anomaly: bool            # True jika bukan "Postur Normal"
    recommendation: str


# ---------------------------------------------------------------------------
# Arsitektur ANN (torch.nn.Module)
# ---------------------------------------------------------------------------

class PostureClassifierANN(nn.Module):
    """
    ANN ringan untuk klasifikasi postur/kelelahan berdasarkan fitur wajah.

    Arsitektur: Input(6) -> Dense(32) -> ReLU -> Dropout(0.3) ->
                Dense(16) -> ReLU -> Dense(5) -> Softmax

    Total parameter: ~700 (sangat ringan, <1MB di memori)
    """

    def __init__(
        self,
        input_size: int = INPUT_FEATURES,
        hidden1_size: int = 32,
        hidden2_size: int = 16,
        num_classes: int = NUM_CLASSES,
        dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            # Layer 1: Input -> Hidden 1
            nn.Linear(input_size, hidden1_size),
            nn.BatchNorm1d(hidden1_size),   # Normalisasi untuk stabilitas training
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            # Layer 2: Hidden 1 -> Hidden 2
            nn.Linear(hidden1_size, hidden2_size),
            nn.ReLU(),

            # Layer 3: Hidden 2 -> Output
            nn.Linear(hidden2_size, num_classes),
        )

        # Inisialisasi bobot dengan Xavier (mencegah vanishing gradient)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Inisialisasi bobot menggunakan Xavier Uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Tensor shape (batch_size, INPUT_FEATURES)

        Returns:
            Logits shape (batch_size, NUM_CLASSES)
        """
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Mengembalikan probabilitas menggunakan Softmax."""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# Feature Extraction
# ---------------------------------------------------------------------------

def extract_features_from_detection(
    face_locations: List[list],
    frame_width: int = 640,
    frame_height: int = 480,
    brightness: float = 128.0,
) -> Optional[np.ndarray]:
    """
    Mengekstrak vektor fitur dari hasil deteksi wajah OpenCV.
    Fitur dinormalisasi ke range [0, 1] untuk kestabilan ANN.

    Args:
        face_locations: List [(x, y, w, h)] dari deteksi Haar Cascade.
        frame_width: Lebar frame kamera.
        frame_height: Tinggi frame kamera.
        brightness: Rata-rata brightness frame.

    Returns:
        numpy array shape (INPUT_FEATURES,) atau None jika tidak ada wajah.
    """
    if not face_locations:
        # Kelas 0: tidak ada wajah
        return np.array([0.0, 0.0, 0.0, 0.0, brightness / 255.0, 0.0], dtype=np.float32)

    # Ambil wajah terbesar (paling dekat ke kamera)
    largest_face = max(face_locations, key=lambda f: f[2] * f[3])
    x, y, w, h = largest_face

    # Normalisasi koordinat ke [0, 1]
    x_center = (x + w / 2) / frame_width       # Posisi horizontal pusat wajah
    y_center = (y + h / 2) / frame_height       # Posisi vertikal pusat wajah
    norm_width = w / frame_width                 # Lebar relatif wajah (indikator jarak)
    norm_height = h / frame_height               # Tinggi relatif wajah
    norm_brightness = brightness / 255.0         # Kecerahan frame
    aspect_ratio = w / max(h, 1)                 # Rasio aspek (bentuk wajah, indikator kemiringan)

    features = np.array(
        [x_center, y_center, norm_width, norm_height, norm_brightness, aspect_ratio],
        dtype=np.float32,
    )

    logger.debug("Fitur diekstrak: %s", features.tolist())
    return features


# ---------------------------------------------------------------------------
# Model Manager
# ---------------------------------------------------------------------------

class ANNModelManager:
    """
    Mengelola siklus hidup model ANN: inisialisasi, inferensi, dan cleanup memori.
    Model dimuat ke device yang sesuai dan dibersihkan setelah inferensi.
    """

    def __init__(self) -> None:
        self._model: Optional[PostureClassifierANN] = None
        self._model_loaded = False

    def _get_model(self) -> PostureClassifierANN:
        """
        Lazy-load model. Muat dari file jika ada, inisialisasi baru jika tidak.
        Model selalu dalam mode eval() untuk inferensi yang konsisten.
        """
        if self._model is None:
            logger.info("Memuat model ANN ke device: %s", DEVICE)
            model = PostureClassifierANN().to(DEVICE)

            if MODEL_SAVE_PATH.exists():
                try:
                    state_dict = torch.load(
                        MODEL_SAVE_PATH,
                        map_location=DEVICE,
                        weights_only=True,
                    )
                    model.load_state_dict(state_dict)
                    logger.info("Bobot model berhasil dimuat dari '%s'.", MODEL_SAVE_PATH)
                    self._model_loaded = True
                except Exception as e:
                    logger.warning(
                        "Gagal memuat bobot dari '%s': %s. "
                        "Menggunakan model dengan bobot awal (random).",
                        MODEL_SAVE_PATH, e,
                    )
            else:
                logger.info(
                    "File bobot model tidak ditemukan di '%s'. "
                    "Menggunakan model baru dengan bobot awal. "
                    "Latih model menggunakan endpoint /api/monitor/train untuk hasil akurat.",
                    MODEL_SAVE_PATH,
                )

            model.eval()  # Mode evaluasi (nonaktifkan dropout)
            self._model = model

        return self._model

    def infer(
        self,
        face_locations: List[list],
        frame_width: int = 640,
        frame_height: int = 480,
        brightness: float = 128.0,
    ) -> InferenceResult:
        """
        Menjalankan inferensi postur berdasarkan fitur wajah yang diekstrak.

        Args:
            face_locations: List [(x, y, w, h)] dari OpenCV.
            frame_width: Lebar frame kamera.
            frame_height: Tinggi frame kamera.
            brightness: Rata-rata brightness frame.

        Returns:
            InferenceResult berisi prediksi dan rekomendasi.
        """
        try:
            # Ekstrak fitur dari deteksi
            features = extract_features_from_detection(
                face_locations=face_locations,
                frame_width=frame_width,
                frame_height=frame_height,
                brightness=brightness,
            )

            if features is None:
                return self._build_no_face_result()

            # Konversi ke tensor PyTorch
            input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            # Inferensi dengan gradient disabled (efisiensi memori)
            model = self._get_model()
            with torch.no_grad():
                probabilities = model.predict_proba(input_tensor)

            # Pindahkan ke CPU untuk pemrosesan
            probs_np = probabilities.cpu().numpy().flatten()

            predicted_class = int(np.argmax(probs_np))
            confidence = float(probs_np[predicted_class])
            predicted_label = CLASS_LABELS.get(predicted_class, "Tidak Diketahui")

            all_probs = {
                CLASS_LABELS[i]: round(float(probs_np[i]), 4)
                for i in range(len(probs_np))
            }

            is_anomaly = predicted_class not in (0, 1)  # 0: no face, 1: normal
            recommendation = self._get_recommendation(predicted_class, confidence)

            result = InferenceResult(
                predicted_class=predicted_class,
                predicted_label=predicted_label,
                confidence=confidence,
                all_probabilities=all_probs,
                is_anomaly=is_anomaly,
                recommendation=recommendation,
            )

            logger.debug(
                "Inferensi: kelas=%d (%s), confidence=%.4f",
                predicted_class, predicted_label, confidence,
            )
            return result

        except Exception as e:
            logger.error("Error saat inferensi ANN: %s — %s", type(e).__name__, e)
            return InferenceResult(
                predicted_class=-1,
                predicted_label="Error Inferensi",
                confidence=0.0,
                all_probabilities={},
                is_anomaly=False,
                recommendation=f"Terjadi kesalahan saat analisis: {e}",
            )
        finally:
            # Bersihkan tensor dari memori GPU/CPU
            gc.collect()
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()

    def _build_no_face_result(self) -> InferenceResult:
        """Membangun hasil untuk kondisi tidak ada wajah terdeteksi."""
        return InferenceResult(
            predicted_class=0,
            predicted_label=CLASS_LABELS[0],
            confidence=1.0,
            all_probabilities={label: 0.0 for label in CLASS_LABELS.values()},
            is_anomaly=False,
            recommendation="Tidak ada wajah terdeteksi. Pastikan kamera menghadap ke Anda.",
        )

    @staticmethod
    def _get_recommendation(predicted_class: int, confidence: float) -> str:
        """Mengembalikan rekomendasi tindakan berdasarkan hasil prediksi."""
        recommendations = {
            0: "Tidak ada wajah terdeteksi. Pastikan Anda berada di depan kamera.",
            1: "Postur Anda baik! Tetap jaga posisi duduk yang nyaman.",
            2: (
                "Terdeteksi tanda kelelahan. Disarankan istirahat 10-15 menit, "
                "regangkan badan, dan minum air putih."
            ),
            3: (
                "Anda terlalu dekat dengan layar. Mundur minimal 50cm untuk "
                "menjaga kesehatan mata."
            ),
            4: (
                "Anda terlalu jauh dari layar atau pencahayaan kurang. "
                "Perbaiki posisi duduk atau tambah pencahayaan."
            ),
        }
        base_recommendation = recommendations.get(predicted_class, "Lanjutkan aktivitas normal.")

        if confidence < 0.5:
            base_recommendation += " (Deteksi kurang akurat, pastikan pencahayaan cukup.)"

        return base_recommendation

    def save_model(self) -> Tuple[bool, str]:
        """
        Menyimpan bobot model ke disk.

        Returns:
            Tuple (sukses: bool, pesan: str)
        """
        if self._model is None:
            return (False, "Model belum dimuat.")

        try:
            MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save(self._model.state_dict(), MODEL_SAVE_PATH)
            msg = f"Bobot model berhasil disimpan ke '{MODEL_SAVE_PATH}'."
            logger.info(msg)
            return (True, msg)
        except Exception as e:
            msg = f"Gagal menyimpan model: {type(e).__name__}: {e}"
            logger.error(msg)
            return (False, msg)

    def get_model_info(self) -> dict:
        """Mengembalikan informasi arsitektur model."""
        model = self._get_model()
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        return {
            "architecture": "PostureClassifierANN",
            "input_features": INPUT_FEATURES,
            "num_classes": NUM_CLASSES,
            "class_labels": CLASS_LABELS,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(DEVICE),
            "model_loaded_from_file": self._model_loaded,
            "model_path": str(MODEL_SAVE_PATH),
        }


# ---------------------------------------------------------------------------
# Singleton Instance
# ---------------------------------------------------------------------------

ann_model_manager = ANNModelManager()
