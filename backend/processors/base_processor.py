from abc import ABC, abstractmethod
from typing import Dict, Any, List
import numpy as np
import cv2


class BaseProcessor(ABC):
    """Базовый класс для всех процессоров изображений"""

    @property
    @abstractmethod
    def id(self) -> str:
        """Уникальный ID процессора (например 'binarization')"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Название процессора для отображения"""
        pass

    @property
    @abstractmethod
    def category(self) -> str:
        """Категория процессора (например 'filter', 'detection')"""
        pass

    @property
    @abstractmethod
    def default_params(self) -> Dict[str, Any]:
        """Параметры по умолчанию"""
        pass

    @property
    @abstractmethod
    def param_definitions(self) -> List[Dict[str, Any]]:
        """Определения параметров для UI"""
        pass

    @abstractmethod
    def process(self, frame: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Применяет обработку к кадру"""
        pass

    def get_param_definitions_for_ui(self) -> List[Dict[str, Any]]:
        """Возвращает определения параметров для UI с доп. информацией"""
        return self.param_definitions

    def _ensure_bgr(self, frame: np.ndarray) -> np.ndarray:
        """Убеждается, что кадр в формате BGR"""
        if len(frame.shape) == 2:
            return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif len(frame.shape) == 3 and frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        return frame

    def _ensure_gray(self, frame: np.ndarray) -> np.ndarray:
        """Убеждается, что кадр в оттенках серого"""
        if len(frame.shape) == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame
