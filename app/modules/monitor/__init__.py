"""Lexa Assistant - Workspace Monitor Module"""
from app.modules.monitor.vision_handler import VisionHandler, vision_handler, DetectionResult
from app.modules.monitor.ann_model import ANNModelManager, ann_model_manager, InferenceResult

__all__ = [
    "VisionHandler",
    "vision_handler",
    "DetectionResult",
    "ANNModelManager",
    "ann_model_manager",
    "InferenceResult",
]
