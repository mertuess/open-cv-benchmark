import cv2
import numpy as np
from typing import Dict, Any, List
from .base_processor import BaseProcessor


class IntensityTransformProcessor(BaseProcessor):
    """Процессор для скалярных (интенсивностных) преобразований"""

    @property
    def id(self) -> str:
        return "intensity_transform"

    @property
    def name(self) -> str:
        return "Скалярные преобразования"

    @property
    def category(self) -> str:
        return "enhancement"

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "enabled": False,
            "transform_type": 0,
            "c": 1.0,
            "gamma": 0.5,
            "scale": 255.0,
            "apply_to_gray": False,
            "normalize_output": True
        }

    @property
    def param_definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": "enabled",
                "label": "Включить",
                "type": "checkbox",
                "default": False
            },
            {
                "name": "transform_type",
                "label": "Тип преобразования",
                "type": "select",
                "options": [
                    {"value": 0, "label": "Негатив (Negative)"},
                    {"value": 1, "label": "Логарифмическое (Log)"},
                    {"value": 2, "label": "Степенное (Gamma)"},
                    {"value": 3,
                        "label": "Обратное логарифмическое (Inverse Log)"}
                ],
                "default": 0
            },
            {
                "name": "c",
                "label": "Константа C",
                "type": "range",
                "min": 1,
                "max": 100,
                "step": 1,
                "default": 10
            },
            {
                "name": "gamma",
                "label": "Гамма (γ)",
                "type": "range",
                "min": 10,
                "max": 500,
                "step": 1,
                "default": 50
            },
            {
                "name": "scale",
                "label": "Масштаб",
                "type": "range",
                "min": 1,
                "max": 255,
                "step": 1,
                "default": 255
            },
            {
                "name": "apply_to_gray",
                "label": "Применять к оттенкам серого",
                "type": "checkbox",
                "default": False
            },
            {
                "name": "normalize_output",
                "label": "Нормализовать выход",
                "type": "checkbox",
                "default": True
            }
        ]

    def process(self, frame: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        if not params.get("enabled", False):
            return frame

        transform_type = int(params.get("transform_type", 0))
        c = float(params.get("c", 10)) / 10.0
        gamma = float(params.get("gamma", 50)) / 100.0
        scale = float(params.get("scale", 255))
        apply_to_gray = bool(params.get("apply_to_gray", False))
        normalize_output = bool(params.get("normalize_output", True))

        try:
            if apply_to_gray:
                gray = self._ensure_gray(frame)
                result_gray = self._apply_transform(
                    gray, transform_type, c, gamma, scale, normalize_output)
                return cv2.cvtColor(result_gray, cv2.COLOR_GRAY2BGR)
            else:
                result = frame.copy()
                for i in range(3):  # BGR
                    channel = result[:, :, i]
                    transformed = self._apply_transform(
                        channel, transform_type, c, gamma, scale, normalize_output)
                    result[:, :, i] = transformed
                return result

        except cv2.error as e:
            print(f"Ошибка в скалярном преобразовании: {e}")
            return frame

    def _apply_transform(self, image: np.ndarray, transform_type: int,
                         c: float, gamma: float, scale: float,
                         normalize: bool) -> np.ndarray:
        """Применяет выбранное преобразование к одноканальному изображению"""

        img_float = image.astype(np.float64)

        if transform_type == 0:  # Негатив
            result = scale - img_float

        elif transform_type == 1:  # Логарифмическое
            result = c * np.log1p(img_float)

        elif transform_type == 2:  # Степенное (Гамма-коррекция)
            if img_float.max() > 1:
                img_normalized = img_float / scale
            else:
                img_normalized = img_float

            result = c * np.power(img_normalized, gamma)

            if img_float.max() > 1:
                result = result * scale

        elif transform_type == 3:  # Обратное логарифмическое
            if img_float.max() > 1:
                img_normalized = img_float / scale
            else:
                img_normalized = img_float

            result = c * (np.exp(img_normalized) - 1)

            if img_float.max() > 1:
                result = result * scale
        else:
            return image

        if normalize:
            if result.max() > 0:
                result = result / result.max() * scale

        result = np.clip(result, 0, scale)

        return result.astype(np.uint8)

    def _visualize_transfer_function(self, transform_type: int,
                                     c: float, gamma: float,
                                     scale: float) -> np.ndarray:
        """Создает визуализацию передаточной функции"""
        x = np.linspace(0, 255, 256, dtype=np.float64)

        if transform_type == 0:  # Негатив
            y = scale - x
        elif transform_type == 1:  # Логарифмическое
            y = c * np.log1p(x)
        elif transform_type == 2:  # Степенное
            x_norm = x / scale
            y = c * np.power(x_norm, gamma) * scale
        elif transform_type == 3:  # Обратное логарифмическое
            x_norm = x / scale
            y = c * (np.exp(x_norm) - 1) * scale
        else:
            y = x

        if y.max() > 0:
            y = y / y.max() * scale

        y = np.clip(y, 0, scale)

        return y.astype(np.uint8)
