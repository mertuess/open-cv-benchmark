import cv2
import numpy as np
from typing import Dict, Any, List
from .base_processor import BaseProcessor


class CornerDetectionProcessor(BaseProcessor):
    """Процессор для детекции углов (Harris, Shi-Tomasi, FAST)"""

    @property
    def id(self) -> str:
        return "corner_detection"

    @property
    def name(self) -> str:
        return "Детекция углов"

    @property
    def category(self) -> str:
        return "feature"

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "method": 0,
            "max_corners": 100,
            "quality_level": 0.01,
            "min_distance": 10,
            "threshold": 0.1,
            "draw_corners": True,
            "corner_color": [0, 0, 255],
            "corner_radius": 3
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
                    {"value": 0, "label": "Harris"},
                    {"value": 1, "label": "Shi-Tomasi"},
                    {"value": 2, "label": "FAST"}
                ],
                "default": 0
            },
            {
                "name": "max_corners",
                "label": "Макс. углов",
                "type": "range",
                "min": 10,
                "max": 500,
                "step": 10,
                "default": 100
            },
            {
                "name": "quality_level",
                "label": "Качество",
                "type": "range",
                "min": 1,
                "max": 100,
                "step": 1,
                "default": 1
            },
            {
                "name": "min_distance",
                "label": "Мин. расстояние",
                "type": "range",
                "min": 1,
                "max": 50,
                "default": 10
            },
            {
                "name": "threshold",
                "label": "Порог",
                "type": "range",
                "min": 1,
                "max": 100,
                "step": 1,
                "default": 10
            },
            {
                "name": "draw_corners",
                "label": "Рисовать углы",
                "type": "checkbox",
                "default": True
            }
        ]

    def process(self, frame: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        if not params.get("enabled", True):
            return frame

        method = int(params.get("method", 0))
        max_corners = int(params.get("max_corners", 100))
        quality_level = float(params.get("quality_level", 1)) / 100.0
        min_distance = int(params.get("min_distance", 10))
        threshold = float(params.get("threshold", 10)) / 100.0
        draw_corners = bool(params.get("draw_corners", True))

        gray = self._ensure_gray(frame)

        corners = []

        if method == 0:  # Harris
            gray_float = np.float32(gray)
            dst = cv2.cornerHarris(gray_float, 2, 3, 0.04)
            dst = cv2.dilate(dst, None)

            threshold_value = threshold * dst.max()
            corner_indices = np.where(dst > threshold_value)
            corners = list(zip(corner_indices[1], corner_indices[0]))

        elif method == 1:  # Shi-Tomasi
            corners_np = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=max_corners,
                qualityLevel=quality_level,
                minDistance=min_distance
            )
            if corners_np is not None:
                corners = [(int(p[0][0]), int(p[0][1])) for p in corners_np]

        elif method == 2:  # FAST
            fast = cv2.FastFeatureDetector_create(
                threshold=int(threshold * 100),
                nonmaxSuppression=True
            )
            keypoints = fast.detect(gray, None)
            corners = [(int(kp.pt[0]), int(kp.pt[1])) for kp in keypoints]

        if draw_corners and corners:
            color = params.get("corner_color", [0, 0, 255])
            radius = int(params.get("corner_radius", 3))

            for (x, y) in corners[:max_corners]:
                cv2.circle(frame, (x, y), radius, color, -1)

        return frame
