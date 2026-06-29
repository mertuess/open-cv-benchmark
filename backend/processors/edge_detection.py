import cv2
import numpy as np
from typing import Dict, Any, List
from .base_processor import BaseProcessor


class EdgeDetectionProcessor(BaseProcessor):
    """Процессор для детекции границ (Canny, Sobel, Laplacian)"""

    @property
    def id(self) -> str:
        return "edge_detection"

    @property
    def name(self) -> str:
        return "Детекция границ"

    @property
    def category(self) -> str:
        return "feature"

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "method": 0,
            "threshold1": 50,
            "threshold2": 150,
            "kernel_size": 3,
            "scale": 1,
            "delta": 0,
            "sobel_dx": 1,
            "sobel_dy": 1,
            "show_edges_only": True,
            "edge_color": [255, 255, 255],
            "background_color": [0, 0, 0]
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
                "name": "method",
                "label": "Метод",
                "type": "select",
                "options": [
                    {"value": 0, "label": "Canny"},
                    {"value": 1, "label": "Sobel"},
                    {"value": 2, "label": "Laplacian"}
                ],
                "default": 0
            },
            {
                "name": "threshold1",
                "label": "Нижний порог",
                "type": "range",
                "min": 0,
                "max": 255,
                "default": 50
            },
            {
                "name": "threshold2",
                "label": "Верхний порог",
                "type": "range",
                "min": 0,
                "max": 255,
                "default": 150
            },
            {
                "name": "kernel_size",
                "label": "Размер ядра",
                "type": "range",
                "min": 1,
                "max": 15,
                "step": 2,
                "default": 3
            },
            {
                "name": "show_edges_only",
                "label": "Показывать только границы",
                "type": "checkbox",
                "default": True
            }
        ]

    def process(self, frame: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        if not params.get("enabled", True):
            return frame

        method = int(params.get("method", 0))
        threshold1 = int(params.get("threshold1", 50))
        threshold2 = int(params.get("threshold2", 150))
        kernel_size = int(params.get("kernel_size", 3))
        scale = float(params.get("scale", 1))
        delta = float(params.get("delta", 0))
        show_edges_only = bool(params.get("show_edges_only", True))

        gray = self._ensure_gray(frame)

        if method == 0:  # Canny
            edges = cv2.Canny(gray, threshold1, threshold2,
                              apertureSize=kernel_size)

        elif method == 1:  # Sobel
            sobel_dx = int(params.get("sobel_dx", 1))
            sobel_dy = int(params.get("sobel_dy", 1))
            edges_x = cv2.Sobel(gray, cv2.CV_64F, sobel_dx,
                                sobel_dy, ksize=kernel_size)
            edges_y = cv2.Sobel(gray, cv2.CV_64F, sobel_dy,
                                sobel_dx, ksize=kernel_size)
            edges = np.sqrt(edges_x**2 + edges_y**2)
            edges = np.uint8(np.clip(edges * scale + delta, 0, 255))

        elif method == 2:  # Laplacian
            edges = cv2.Laplacian(gray, cv2.CV_64F, ksize=kernel_size)
            edges = np.uint8(np.clip(np.abs(edges) * scale + delta, 0, 255))
        else:
            return frame

        if show_edges_only:
            result = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        else:
            edge_color = params.get("edge_color", [255, 255, 255])
            result = frame.copy()

            mask = edges > 0
            for c in range(3):
                result[:, :, c][mask] = edge_color[c]

        return result
