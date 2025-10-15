# src/validation/base.py
from abc import ABC, abstractmethod
from typing import Iterator, Tuple
import pandas as pd
import numpy as np

class BaseSplitter(ABC):
    """
    Абстрактный базовый класс для всех стратегий разделения данных.
    """
    @abstractmethod
    def split(self, data: pd.DataFrame, y: pd.Series, groups: pd.Series = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Генерирует индексы для разделения данных на train и validation.

        Args:
            data (pd.DataFrame): Полный обучающий DataFrame, содержащий все
                                 необходимые колонки (например, дату, группы).
            y (pd.Series): Целевая переменная.
            groups (pd.Series, optional): Группы для групповой валидации.

        Yields:
            Iterator[Tuple[np.ndarray, np.ndarray]]: Кортеж с train и validation индексами.
        """
        pass

    @abstractmethod
    def get_n_splits(self, data: pd.DataFrame = None, y: pd.Series = None, groups: pd.Series = None) -> int:
        """
        Возвращает количество фолдов, создаваемых этим сплиттером.
        """
        pass