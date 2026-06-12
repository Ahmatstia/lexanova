"""Lexa Assistant - Workspace Monitor Module"""
# Import di-lazy agar tidak memblock startup server
# torch dan cv2 dimuat hanya saat dibutuhkan

__all__ = [
    "VisionHandler",
    "vision_handler",
    "DetectionResult",
    "ANNModelManager",
    "ann_model_manager",
    "InferenceResult",
]
