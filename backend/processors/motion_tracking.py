import cv2
import numpy as np
from typing import Dict, Any, List, Tuple
from collections import deque
from .base_processor import BaseProcessor


class MotionTrackingProcessor(BaseProcessor):
    """Процессор для отслеживания движения лиц"""

    def __init__(self):
        self.face_history: Dict[int, deque] = {}
        self.next_id = 0
        self.frame_count = 0
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    @property
    def id(self) -> str:
        return "motion_tracking"

    @property
    def name(self) -> str:
        return "Отслеживание движения"

    @property
    def category(self) -> str:
        return "detection"

    @property
    def default_params(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "history_size": 10,
            "movement_threshold": 5.0,
            "scaleFactor": 1.1,
            "minNeighbors": 5,
            "minSize": 30,
            "draw_trajectory": True,
            "trajectory_length": 10,
            "trajectory_color": [255, 255, 0],
            "moving_color": [0, 255, 0],
            "stationary_color": [0, 165, 255],
            "show_status_text": True,
            "show_id": True,
            "rect_thickness": 2,
            "detection_interval": 2
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
                "name": "history_size",
                "label": "Размер истории (кадров)",
                "type": "range",
                "min": 3,
                "max": 30,
                "step": 1,
                "default": 10
            },
            {
                "name": "movement_threshold",
                "label": "Порог движения (пикс.)",
                "type": "range",
                "min": 1,
                "max": 50,
                "step": 1,
                "default": 5
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
                "label": "Мин. размер лица",
                "type": "range",
                "min": 10,
                "max": 100,
                "default": 30
            },
            {
                "name": "draw_trajectory",
                "label": "Рисовать траекторию",
                "type": "checkbox",
                "default": True
            },
            {
                "name": "show_status_text",
                "label": "Показывать статус",
                "type": "checkbox",
                "default": True
            },
            {
                "name": "show_id",
                "label": "Показывать ID",
                "type": "checkbox",
                "default": True
            },
            {
                "name": "detection_interval",
                "label": "Интервал детекции",
                "type": "range",
                "min": 1,
                "max": 10,
                "step": 1,
                "default": 2
            }
        ]

    def process(self, frame: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        if not params.get("enabled", True):
            return frame

        self.frame_count += 1

        history_size = int(params.get("history_size", 10))
        movement_threshold = float(params.get("movement_threshold", 5.0))
        scaleFactor = float(params.get("scaleFactor", 110)) / 100.0
        minNeighbors = int(params.get("minNeighbors", 5))
        minSize = int(params.get("minSize", 30))
        draw_trajectory = bool(params.get("draw_trajectory", True))
        show_status_text = bool(params.get("show_status_text", True))
        show_id = bool(params.get("show_id", True))
        detection_interval = int(params.get("detection_interval", 2))

        moving_color = params.get("moving_color", [0, 255, 0])
        stationary_color = params.get("stationary_color", [0, 165, 255])
        trajectory_color = params.get("trajectory_color", [255, 255, 0])
        trajectory_length = int(params.get("trajectory_length", 10))
        rect_thickness = int(params.get("rect_thickness", 2))

        result = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.frame_count % detection_interval == 0:
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=scaleFactor,
                minNeighbors=minNeighbors,
                minSize=(minSize, minSize)
            )

            self._update_trackers(faces, history_size)

        self._draw_tracked_faces(
            result,
            history_size,
            movement_threshold,
            moving_color,
            stationary_color,
            draw_trajectory,
            trajectory_color,
            trajectory_length,
            show_status_text,
            show_id,
            rect_thickness
        )

        self._draw_info(result, history_size)

        return result

    def _update_trackers(self, faces: List[Tuple[int, int, int, int]], history_size: int):
        """Обновляет трекеры лиц"""
        detected_centers = []
        for (x, y, w, h) in faces:
            center_x = x + w // 2
            center_y = y + h // 2
            detected_centers.append((center_x, center_y, x, y, w, h))

        if not detected_centers:
            for face_id in list(self.face_history.keys()):
                if len(self.face_history[face_id]) > 0:
                    last_pos = self.face_history[face_id][-1]
                    self.face_history[face_id].append(last_pos)
                    while len(self.face_history[face_id]) > history_size:
                        self.face_history[face_id].popleft()
            return

        matched_ids = set()
        unused_ids = set(self.face_history.keys())

        for center_x, center_y, x, y, w, h in detected_centers:
            matched = False
            min_distance = float('inf')
            best_id = None

            for face_id in unused_ids:
                if len(self.face_history[face_id]) > 0:
                    last_pos = self.face_history[face_id][-1]
                    last_x, last_y, _, _, _, _ = last_pos
                    distance = np.sqrt((center_x - last_x) **
                                       2 + (center_y - last_y)**2)

                    if distance < min_distance and distance < 100:
                        min_distance = distance
                        best_id = face_id

            if best_id is not None:
                self.face_history[best_id].append(
                    (center_x, center_y, x, y, w, h))
                while len(self.face_history[best_id]) > history_size:
                    self.face_history[best_id].popleft()
                matched_ids.add(best_id)
                unused_ids.remove(best_id)
                matched = True

            if not matched:
                new_id = self.next_id
                self.next_id += 1
                self.face_history[new_id] = deque(maxlen=history_size)
                for _ in range(min(5, history_size)):
                    self.face_history[new_id].append(
                        (center_x, center_y, x, y, w, h))
                matched_ids.add(new_id)

        for face_id in list(self.face_history.keys()):
            if face_id not in matched_ids:
                if len(self.face_history[face_id]) > 0:
                    del self.face_history[face_id]

    def _draw_tracked_faces(self, frame: np.ndarray, history_size: int,
                            movement_threshold: float, moving_color: List[int],
                            stationary_color: List[int], draw_trajectory: bool,
                            trajectory_color: List[int], trajectory_length: int,
                            show_status_text: bool, show_id: bool,
                            rect_thickness: int):
        """Отрисовывает отслеживаемые лица"""

        for face_id, history in self.face_history.items():
            if len(history) < 2:
                continue

            current_pos = history[-1]
            center_x, center_y, x, y, w, h = current_pos

            is_moving = self._is_moving(history, movement_threshold)

            color = moving_color if is_moving else stationary_color

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, rect_thickness)

            cv2.circle(frame, (center_x, center_y), 3, color, -1)

            if show_id:
                cv2.putText(
                    frame,
                    f"ID: {face_id}",
                    (x, y - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1
                )

            if show_status_text:
                status = "MOVING" if is_moving else "STATIONARY"
                status_color = moving_color if is_moving else stationary_color
                cv2.putText(
                    frame,
                    status,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    status_color,
                    1
                )

            if draw_trajectory and len(history) > 1:
                points = []
                start_idx = max(0, len(history) - trajectory_length)
                for i in range(start_idx, len(history)):
                    pos = history[i]
                    px, py, _, _, _, _ = pos
                    points.append((px, py))

                if len(points) > 1:
                    for i in range(len(points) - 1):
                        cv2.line(
                            frame,
                            points[i],
                            points[i + 1],
                            trajectory_color,
                            1
                        )

                for i, (px, py) in enumerate(points):
                    alpha = i / len(points)
                    color_intensity = int(255 * alpha)
                    point_color = [color_intensity, color_intensity, 255]
                    cv2.circle(frame, (px, py), 2, point_color, -1)

    def _is_moving(self, history: deque, movement_threshold: float) -> bool:
        """Определяет, движется ли лицо на основе истории позиций"""
        if len(history) < 2:
            return False

        movements = []
        for i in range(1, min(len(history), 5)):
            prev_pos = history[-i-1]
            curr_pos = history[-i]
            prev_x, prev_y, _, _, _, _ = prev_pos
            curr_x, curr_y, _, _, _, _ = curr_pos

            distance = np.sqrt((curr_x - prev_x)**2 + (curr_y - prev_y)**2)
            movements.append(distance)

        if not movements:
            return False

        avg_movement = np.mean(movements)
        return avg_movement > movement_threshold

    def _draw_info(self, frame: np.ndarray, history_size: int):
        """Рисует информационную панель"""
        face_count = len(self.face_history)

        cv2.putText(
            frame,
            f"Tracking: {face_count} faces",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            [255, 255, 255],
            2
        )
