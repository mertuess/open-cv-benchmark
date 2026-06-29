from typing import List
from .base_pipeline import BasePipeline


class SimplePipeline(BasePipeline):
    """Простой пайплайн - только бинаризация"""

    @property
    def id(self) -> str:
        return "simple"

    @property
    def name(self) -> str:
        return "Простая бинаризация"

    @property
    def description(self) -> str:
        return "Только бинаризация"

    def get_processor_chain(self) -> List[str]:
        return ["binarization"]


class IntensityPipeline(BasePipeline):
    """Пайплайн со скалярными преобразованиями"""

    @property
    def id(self) -> str:
        return "intensity"

    @property
    def name(self) -> str:
        return "Скалярные преобразования"

    @property
    def description(self) -> str:
        return "Негатив, логарифмическое, степенное, обратное логарифмическое"

    def get_processor_chain(self) -> List[str]:
        return ["intensity_transform"]


class BlurPipeline(BasePipeline):
    """Пайплайн с сглаживанием"""

    @property
    def id(self) -> str:
        return "blur"

    @property
    def name(self) -> str:
        return "Сглаживание"

    @property
    def description(self) -> str:
        return "Пространственное сглаживание (размытие)"

    def get_processor_chain(self) -> List[str]:
        return ["blur"]


class FaceDetectionPipeline(BasePipeline):
    """Пайплайн для детекции лиц"""

    @property
    def id(self) -> str:
        return "face_detection"

    @property
    def name(self) -> str:
        return "Детекция лиц"

    @property
    def description(self) -> str:
        return "Обнаружение лиц, глаз и улыбок"

    def get_processor_chain(self) -> List[str]:
        return ["face_detection"]


class EdgeDetectionPipeline(BasePipeline):
    """Пайплайн для детекции границ"""

    @property
    def id(self) -> str:
        return "edge_detection"

    @property
    def name(self) -> str:
        return "Детекция границ"

    @property
    def description(self) -> str:
        return "Обнаружение границ (Canny, Sobel, Laplacian)"

    def get_processor_chain(self) -> List[str]:
        return ["edge_detection"]


class CornerDetectionPipeline(BasePipeline):
    """Пайплайн для детекции углов"""

    @property
    def id(self) -> str:
        return "corner_detection"

    @property
    def name(self) -> str:
        return "Детекция углов"

    @property
    def description(self) -> str:
        return "Обнаружение углов (Harris, Shi-Tomasi, FAST)"

    def get_processor_chain(self) -> List[str]:
        return ["corner_detection"]


class ContourDetectionPipeline(BasePipeline):
    """Пайплайн для поиска контуров"""

    @property
    def id(self) -> str:
        return "contour_detection"

    @property
    def name(self) -> str:
        return "Поиск контуров"

    @property
    def description(self) -> str:
        return "Поиск и отрисовка контуров"

    def get_processor_chain(self) -> List[str]:
        return ["contour_detection"]


class PreprocessingPipeline(BasePipeline):
    """Пайплайн предобработки: сглаживание + бинаризация"""

    @property
    def id(self) -> str:
        return "preprocessing"

    @property
    def name(self) -> str:
        return "Предобработка"

    @property
    def description(self) -> str:
        return "Сглаживание -> Бинаризация"

    def get_processor_chain(self) -> List[str]:
        return ["blur", "binarization"]


class IntensityEdgePipeline(BasePipeline):
    """Пайплайн: скалярные преобразования + границы"""

    @property
    def id(self) -> str:
        return "intensity_edge"

    @property
    def name(self) -> str:
        return "Интенсивность + Границы"

    @property
    def description(self) -> str:
        return "Скалярные преобразования с последующей детекцией границ"

    def get_processor_chain(self) -> List[str]:
        return ["intensity_transform", "edge_detection"]


class EdgeContourPipeline(BasePipeline):
    """Пайплайн: границы + контуры"""

    @property
    def id(self) -> str:
        return "edge_contour"

    @property
    def name(self) -> str:
        return "Границы и контуры"

    @property
    def description(self) -> str:
        return "Сначала границы, затем контуры"

    def get_processor_chain(self) -> List[str]:
        return ["edge_detection", "contour_detection"]


class FullPipeline(BasePipeline):
    """Полный пайплайн: все методы"""

    @property
    def id(self) -> str:
        return "full"

    @property
    def name(self) -> str:
        return "Полный пайплайн"

    @property
    def description(self) -> str:
        return "Сглаживание -> Интенсивность -> Бинаризация -> Границы -> Контуры -> Углы -> Лица"

    def get_processor_chain(self) -> List[str]:
        return [
            "blur",
            "intensity_transform",
            "binarization",
            "edge_detection",
            "contour_detection",
            "corner_detection",
            "face_detection"
        ]


class MotionTrackingPipeline(BasePipeline):
    """Пайплайн для отслеживания движения"""

    @property
    def id(self) -> str:
        return "motion_tracking"

    @property
    def name(self) -> str:
        return "Отслеживание движения"

    @property
    def description(self) -> str:
        return "Отслеживание движения лиц (зеленый - движется, оранжевый - покоится)"

    def get_processor_chain(self) -> List[str]:
        return ["motion_tracking"]


ALL_PIPELINES = [
    SimplePipeline(),
    IntensityPipeline(),
    BlurPipeline(),
    FaceDetectionPipeline(),
    EdgeDetectionPipeline(),
    CornerDetectionPipeline(),
    ContourDetectionPipeline(),
    PreprocessingPipeline(),
    IntensityEdgePipeline(),
    EdgeContourPipeline(),
    MotionTrackingPipeline(),
    FullPipeline(),
]
