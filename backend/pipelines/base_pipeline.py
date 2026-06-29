from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np
from processors.base_processor import BaseProcessor


class BasePipeline(ABC):
    """Базовый класс для пайплайнов обработки"""

    @property
    @abstractmethod
    def id(self) -> str:
        """Уникальный ID пайплайна"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Название пайплайна для отображения"""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Описание пайплайна"""
        pass

    @abstractmethod
    def get_processor_chain(self) -> List[str]:
        """Возвращает список ID процессоров в порядке применения"""
        pass

    def get_all_params(self, processors: Dict[str, BaseProcessor]) -> Dict[str, Dict[str, Any]]:
        """Собирает параметры всех процессоров в пайплайне"""
        params = {}
        for processor_id in self.get_processor_chain():
            if processor_id in processors:
                params[processor_id] = processors[processor_id].default_params.copy()
        return params

    def process(self, frame: np.ndarray,
                processors: Dict[str, BaseProcessor],
                params: Dict[str, Dict[str, Any]]) -> np.ndarray:
        """Применяет все процессоры в цепочке"""
        result = frame.copy()

        for processor_id in self.get_processor_chain():
            if processor_id in processors:
                processor = processors[processor_id]
                processor_params = params.get(
                    processor_id, processor.default_params)
                result = processor.process(result, processor_params)

        return result
