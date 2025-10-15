# src/models/base.py

from abc import ABC, abstractmethod
from typing import Any
import pandas as pd

class ModelInterface(ABC):
    """
    Абстрактный базовый класс (интерфейс) для всех моделей.
    """
    
    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> None:
        """Обучает модель."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> Any:
        """Возвращает предсказания классов."""
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> Any:
        """Возвращает предсказания вероятностей."""
        pass

    @abstractmethod
    def save(self, filepath: str) -> None:
        """Сохраняет модель в файл."""
        pass

    @classmethod
    @abstractmethod
    def load(cls, filepath: str) -> 'ModelInterface':
        """Загружает модель из файла."""
        pass