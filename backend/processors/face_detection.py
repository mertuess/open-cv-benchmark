import cv2
import numpy as np
from typing import Dict, Any, List
from .base_processor import BaseProcessor


class FaceDetectionProcessor(BaseProcessor):
    """Процессор для детекции лиц с использованием каскадов Хаара"""

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.profile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_profileface.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        self.smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_smile.xml'
        )

    @property
    def id(self) -> str:
        return "face_detection"

    @property
    def name(self) -> str:
        return "Детекция лиц"

    @property
    def category(self) -> str:
        return "detection"

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "detection_type": 0,
            "scaleFactor": 1.1,
            "minNeighbors": 5,
            "minSize": 30,
            "maxSize": 500,
            "draw_rectangles": True,
            "rect_color": [0, 255, 0],
            "rect_thickness": 2
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
                "name": "detection_type",
                "label": "Тип детекции",
                "type": "select",
                "options": [
                    {"value": 0, "label": "Лица"},
                    {"value": 1, "label": "Глаза"},
                    {"value": 2, "label": "Улыбки"},
                    {"value": 3, "label": "Все"}
                ],
                "default": 0
            },
            {
                "name": "scaleFactor",
                "label": "Масштабный фактор",
                "type": "range",
                "min": 101,
                "max": 200,
                "step": 1,
                "default": 110
            },
            {
                "name": "minNeighbors",
                "label": "Мин. соседей",
                "type": "range",
                "min": 1,
                "max": 10,
                "default": 5
            },
            {
                "name": "minSize",
                "label": "Мин. размер",
                "type": "range",
                "min": 10,
                "max": 100,
                "default": 30
            },
            {
                "name": "maxSize",
                "label": "Макс. размер",
                "type": "range",
                "min": 100,
                "max": 1000,
                "step": 10,
                "default": 500
            },
            {
                "name": "draw_rectangles",
                "label": "Рисовать рамки",
                "type": "checkbox",
                "default": True
            }
        ]

    def process(self, frame: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        if not params.get("enabled", True):
            return frame

        detection_type = int(params.get("detection_type", 0))
        scaleFactor = float(params.get("scaleFactor", 110)) / 100.0
        minNeighbors = int(params.get("minNeighbors", 5))
        minSize = int(params.get("minSize", 30))
        maxSize = int(params.get("maxSize", 500))
        draw_rectangles = bool(params.get("draw_rectangles", True))

        gray = self._ensure_gray(frame)

        result = frame.copy() if draw_rectangles else frame

        if detection_type == 0 or detection_type == 3:
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=scaleFactor,
                minNeighbors=minNeighbors,
                minSize=(minSize, minSize),
                maxSize=(maxSize, maxSize)
            )
            if draw_rectangles:
                for (x, y, w, h) in faces:
                    cv2.rectangle(result, (x, y), (x+w, y+h), [0, 255, 0], 2)
                    cv2.putText(result, "FACE", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 0], 1)

        if detection_type == 1 or detection_type == 3:
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 5)
                if draw_rectangles:
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(result, (x+ex, y+ey), (x+ex+ew, y+ey+eh),
                                      [0, 0, 255], 2)
                        cv2.putText(result, "EYE", (x+ex, y+ey-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255], 1)

        if detection_type == 2 or detection_type == 3:
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                smiles = self.smile_cascade.detectMultiScale(roi_gray, 1.1, 5)
                if draw_rectangles:
                    for (sx, sy, sw, sh) in smiles:
                        cv2.rectangle(result, (x+sx, y+sy), (x+sx+sw, y+sy+sh),
                                      [255, 0, 0], 2)
                        cv2.putText(result, "SMILE", (x+sx, y+sy-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, [255, 0, 0], 1)

        return result
