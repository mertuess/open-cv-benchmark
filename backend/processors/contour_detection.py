import cv2
import numpy as np
from typing import Dict, Any, List
from .base_processor import BaseProcessor


class ContourDetectionProcessor(BaseProcessor):
    """Процессор для поиска и отрисовки контуров"""

    @property
    def id(self) -> str:
        return "contour_detection"

    @property
    def name(self) -> str:
        return "Поиск контуров"

    @property
    def category(self) -> str:
        return "feature"

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "threshold_low": 50,
            "threshold_high": 150,
            "mode": 0,
            "method": 0,
            "draw_contours": True,
            "contour_color": [0, 255, 0],
            "contour_thickness": 2,
            "fill_contours": False,
            "min_area": 100,
            "max_area": 10000
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
                "name": "threshold_low",
                "label": "Нижний порог",
                "type": "range",
                "min": 0,
                "max": 255,
                "default": 50
            },
            {
                "name": "threshold_high",
                "label": "Верхний порог",
                "type": "range",
                "min": 0,
                "max": 255,
                "default": 150
            },
            {
                "name": "mode",
                "label": "Режим поиска",
                "type": "select",
                "options": [
                    {"value": 0, "label": "Только внешние"},
                    {"value": 1, "label": "Все контуры"},
                    {"value": 2, "label": "С иерархией"}
                ],
                "default": 0
            },
            {
                "name": "method",
                "label": "Метод аппроксимации",
                "type": "select",
                "options": [
                    {"value": 0, "label": "Простая"},
                    {"value": 1, "label": "Полная"}
                ],
                "default": 0
            },
            {
                "name": "draw_contours",
                "label": "Рисовать контуры",
                "type": "checkbox",
                "default": True
            },
            {
                "name": "fill_contours",
                "label": "Заполнять контуры",
                "type": "checkbox",
                "default": False
            },
            {
                "name": "min_area",
                "label": "Мин. площадь",
                "type": "range",
                "min": 10,
                "max": 5000,
                "step": 10,
                "default": 100
            },
            {
                "name": "max_area",
                "label": "Макс. площадь",
                "type": "range",
                "min": 100,
                "max": 50000,
                "step": 100,
                "default": 10000
            }
        ]

    def process(self, frame: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        if not params.get("enabled", True):
            return frame

        threshold_low = int(params.get("threshold_low", 50))
        threshold_high = int(params.get("threshold_high", 150))
        mode = int(params.get("mode", 0))
        method = int(params.get("method", 0))
        draw_contours = bool(params.get("draw_contours", True))
        fill_contours = bool(params.get("fill_contours", False))
        min_area = int(params.get("min_area", 100))
        max_area = int(params.get("max_area", 10000))

        gray = self._ensure_gray(frame)

        edges = cv2.Canny(gray, threshold_low, threshold_high)

        if mode == 0:
            retrieval_mode = cv2.RETR_EXTERNAL
        elif mode == 1:
            retrieval_mode = cv2.RETR_LIST
        else:
            retrieval_mode = cv2.RETR_TREE

        if method == 0:
            approx_method = cv2.CHAIN_APPROX_SIMPLE
        else:
            approx_method = cv2.CHAIN_APPROX_NONE

        contours, hierarchy = cv2.findContours(
            edges, retrieval_mode, approx_method
        )

        filtered_contours = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area <= area <= max_area:
                filtered_contours.append(cnt)

        if draw_contours and filtered_contours:
            color = params.get("contour_color", [0, 255, 0])
            thickness = int(params.get("contour_thickness", 2))

            if fill_contours:
                cv2.drawContours(frame, filtered_contours, -1, color, -1)
            else:
                cv2.drawContours(frame, filtered_contours, -
                                 1, color, thickness)

            cv2.putText(
                frame,
                f"Contours: {len(filtered_contours)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                [255, 255, 255],
                2
            )

        return frame
