# src/features/base.py
from abc import ABC, abstractmethod
import pandas as pd
from typing import Literal

# Определяем возможные стратегии обучения
FitStrategy = Literal["train_only", "combined"]

class FeatureGenerator(ABC):
    """
    Абстрактный базовый класс для всех генераторов признаков.
    """
    # По умолчанию - самая безопасная стратегия.
    fit_strategy: FitStrategy = "train_only"
    
    def __init__(self, name: str, **kwargs):
        self.name = name

    @abstractmethod
    def fit(self, data: pd.DataFrame) -> None:
        """
        Обучает генератор на данных (например, вычисляет средние, медианы,
        создает словарь для кодирования).
        Этот метод вызывается ТОЛЬКО на обучающей выборке.
        """
        pass

    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Применяет преобразование к данным, создавая новые признаки.
        """
        pass

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Выполняет fit, а затем transform.
        """
        self.fit(data)
        return self.transform(data)