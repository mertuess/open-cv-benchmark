from typing import Dict, List, Optional
from .base_processor import BaseProcessor
from .binarization import BinarizationProcessor
from .face_detection import FaceDetectionProcessor
from .corner_detection import CornerDetectionProcessor
from .contour_detection import ContourDetectionProcessor
from .edge_detection import EdgeDetectionProcessor
from .blur import BlurProcessor
from .intensity_transfrom import IntensityTransformProcessor
from .motion_tracking import MotionTrackingProcessor


class ProcessorManager:
    """Менеджер для управления всеми процессорами"""

    def __init__(self):
        self._processors: Dict[str, BaseProcessor] = {}
        self._register_processors()

    def _register_processors(self):
        """Регистрирует все доступные процессоры"""
        processors = [
            BinarizationProcessor(),
            FaceDetectionProcessor(),
            CornerDetectionProcessor(),
            ContourDetectionProcessor(),
            EdgeDetectionProcessor(),
            BlurProcessor(),
            IntensityTransformProcessor(),
            MotionTrackingProcessor(),
        ]

        for processor in processors:
            self._processors[processor.id] = processor

    def get_processor(self, processor_id: str) -> Optional[BaseProcessor]:
        """Возвращает процессор по ID"""
        return self._processors.get(processor_id)

    def get_all_processors(self) -> List[BaseProcessor]:
        """Возвращает список всех процессоров"""
        return list(self._processors.values())

    def get_processors_by_category(self, category: str) -> List[BaseProcessor]:
        """Возвращает процессоры по категории"""
        return [p for p in self._processors.values() if p.category == category]


processor_manager = ProcessorManager()
