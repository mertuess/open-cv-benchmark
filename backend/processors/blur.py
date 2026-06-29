import cv2
import numpy as np
from typing import Dict, Any, List
from .base_processor import BaseProcessor


class BlurProcessor(BaseProcessor):
    """Процессор для пространственного сглаживания (размытия)"""

    @property
    def id(self) -> str:
        return "blur"

    @property
    def name(self) -> str:
        return "Пространственное сглаживание"

    @property
    def category(self) -> str:
        return "filter"

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "enabled": False,
            "blur_type": 0,
            "kernel_size": 5,
            "sigma_x": 0,
            "sigma_y": 0,
            "sigma_color": 75,
            "sigma_space": 75,
            "border_type": 0,
            "normalize": True
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
                "name": "blur_type",
                "label": "Тип сглаживания",
                "type": "select",
                "options": [
                    {"value": 0, "label": "Простое (Average)"},
                    {"value": 1, "label": "Гауссово (Gaussian)"},
                    {"value": 2, "label": "Медианное (Median)"},
                    {"value": 3, "label": "Билатеральное (Bilateral)"},
                    {"value": 4, "label": "Box фильтр"}
                ],
                "default": 0
            },
            {
                "name": "kernel_size",
                "label": "Размер ядра",
                "type": "range",
                "min": 1,
                "max": 31,
                "step": 2,
                "default": 5
            },
            {
                "name": "sigma_x",
                "label": "Сигма по X",
                "type": "range",
                "min": 0,
                "max": 100,
                "step": 1,
                "default": 0
            },
            {
                "name": "sigma_y",
                "label": "Сигма по Y",
                "type": "range",
                "min": 0,
                "max": 100,
                "step": 1,
                "default": 0
            },
            {
                "name": "sigma_color",
                "label": "Сигма Цвет (Билатериальный)",
                "type": "range",
                "min": 1,
                "max": 150,
                "step": 1,
                "default": 75
            },
            {
                "name": "sigma_space",
                "label": "Сигма пространство (Билатериальный)",
                "type": "range",
                "min": 1,
                "max": 150,
                "step": 1,
                "default": 75
            },
            {
                "name": "border_type",
                "label": "Тип границ",
                "type": "select",
                "options": [
                    {"value": 0, "label": "DEFAULT"},
                    {"value": 1, "label": "CONSTANT"},
                    {"value": 2, "label": "REPLICATE"},
                    {"value": 3, "label": "REFLECT"},
                    {"value": 4, "label": "WRAP"},
                    {"value": 5, "label": "REFLECT101"}
                ],
                "default": 0
            },
            {
                "name": "normalize",
                "label": "Нормализовать",
                "type": "checkbox",
                "default": True
            }
        ]

    def process(self, frame: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        if not params.get("enabled", False):
            return frame

        blur_type = int(params.get("blur_type", 0))
        kernel_size = int(params.get("kernel_size", 5))
        sigma_x = float(params.get("sigma_x", 0))
        sigma_y = float(params.get("sigma_y", 0))
        sigma_color = float(params.get("sigma_color", 75))
        sigma_space = float(params.get("sigma_space", 75))
        border_type = int(params.get("border_type", 0))
        normalize = bool(params.get("normalize", True))

        kernel_size = self._normalize_kernel_size(kernel_size)

        if kernel_size == 1 and blur_type != 2:
            return frame

        border_modes = {
            0: cv2.BORDER_DEFAULT,
            1: cv2.BORDER_CONSTANT,
            2: cv2.BORDER_REPLICATE,
            3: cv2.BORDER_REFLECT,
            4: cv2.BORDER_WRAP,
            5: cv2.BORDER_REFLECT101
        }
        border_mode = border_modes.get(border_type, cv2.BORDER_DEFAULT)

        try:
            if blur_type == 0:  # Простое сглаживание (Average)
                result = cv2.blur(frame, (kernel_size, kernel_size),
                                  borderType=border_mode)

            elif blur_type == 1:  # Гауссово сглаживание
                if sigma_x == 0 and sigma_y == 0:
                    sigma_x = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
                    sigma_y = sigma_x
                result = cv2.GaussianBlur(frame, (kernel_size, kernel_size),
                                          sigma_x, sigma_y, borderType=border_mode)

            elif blur_type == 2:  # Медианное сглаживание
                if kernel_size < 3:
                    kernel_size = 3
                if kernel_size % 2 == 0:
                    kernel_size += 1
                result = cv2.medianBlur(frame, kernel_size)

            elif blur_type == 3:  # Билатеральное сглаживание
                d = kernel_size
                if d < 5:
                    d = 5
                if d % 2 == 0:
                    d += 1
                result = cv2.bilateralFilter(frame, d, sigma_color, sigma_space,
                                             borderType=border_mode)

            elif blur_type == 4:  # Box фильтр
                result = cv2.boxFilter(frame, -1, (kernel_size, kernel_size),
                                       normalize=normalize, borderType=border_mode)
            else:
                return frame

            return result

        except cv2.error as e:
            print(f"Ошибка в сглаживании: {e}")
            return frame

    @staticmethod
    def _normalize_kernel_size(kernel_size: int) -> int:
        """Нормализует размер ядра (нечетное число)"""
        kernel_size = int(kernel_size)
        if kernel_size < 1:
            kernel_size = 1
        if kernel_size % 2 == 0:
            kernel_size += 1
        return kernel_size

    @staticmethod
    def _create_kernel(kernel_size: int, sigma: float = 0) -> np.ndarray:
        """Создает ядро Гаусса для визуализации"""
        if sigma == 0:
            sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
        kernel = cv2.getGaussianKernel(kernel_size, sigma)
        kernel_2d = np.outer(kernel, kernel)
        kernel_2d = kernel_2d / kernel_2d.max()
        return kernel_2d
