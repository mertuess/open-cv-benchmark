import cv2
import numpy as np
from typing import Dict, Any, List
from .base_processor import BaseProcessor


class BinarizationProcessor(BaseProcessor):
    """Процессор для бинаризации изображений"""

    @property
    def id(self) -> str:
        return "binarization"

    @property
    def name(self) -> str:
        return "Бинаризация"

    @property
    def category(self) -> str:
        return "filter"

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "binarizationType": 1,
            "bin_thresh_t": 127,
            "win_size": 50,
            "c_val": 10.0
        }

    @property
    def param_definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "enabled",
                "label": "Включить",
                "type": "checkbox",
                "default": True
            },
            {
                "name": "binarizationType",
                "label": "Тип бинаризации",
                "type": "select",
                "options": [
                    {"value": 1, "label": "Глобальная"},
                    {"value": 2, "label": "Метод Отсу"},
                    {"value": 3, "label": "Среднее арифметическое"},
                    {"value": 4, "label": "Адапт. по весам Гаусса"}
                ],
                "default": 1
            },
            {
                "name": "bin_thresh_t",
                "label": "Отсечение яркости",
                "type": "range",
                "min": 0,
                "max": 255,
                "default": 127
            },
            {
                "name": "win_size",
                "label": "Размер окна",
                "type": "range",
                "min": 3,
                "max": 511,
                "step": 2,
                "default": 50
            },
            {
                "name": "c_val",
                "label": "Сдвиг локал. порога",
                "type": "range",
                "min": 0,
                "max": 255,
                "step": 1,
                "default": 10
            }
        ]

    def process(self, frame: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        if not params.get("enabled", True):
            return frame

        method_id = int(params.get("binarizationType", 1))

        if frame is None or frame.size == 0:
            return frame

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except cv2.error:
            gray = frame

        if method_id == 1:  # Глобальная бинаризация
            thresh = float(params.get("bin_thresh_t", 127))
            _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)

        elif method_id == 2:  # Метод Отсу
            _, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        elif method_id == 3:  # Адаптивная (среднее арифметическое)
            win_size = int(params.get("win_size", 50))
            c_val = float(params.get("c_val", 10.0))
            win_size = self._normalize_win_size(win_size)

            binary = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                win_size,
                c_val
            )

        elif method_id == 4:  # Адаптивная (по весам Гаусса)
            win_size = int(params.get("win_size", 50))
            c_val = float(params.get("c_val", 10.0))
            win_size = self._normalize_win_size(win_size)

            binary = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                win_size,
                c_val
            )
        else:
            return frame

        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _normalize_win_size(win_size: int) -> int:
        """Нормализует размер окна (нечетное и >= 3)"""
        win_size = int(win_size)

        if win_size < 3:
            win_size = 3
        elif win_size % 2 == 0:
            win_size += 1

        if win_size > 511:
            win_size = 511

        return win_size
