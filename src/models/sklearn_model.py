# src/models/sklearn_model.py

from typing import Any, Dict

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression # Пример, будет заменен Hydra
from sklearn.base import BaseEstimator

from .base import ModelInterface

# ==================================================================================
# SklearnModel
# ==================================================================================
class SklearnModel(ModelInterface):
    """
    Универсальный класс-обертка для моделей из библиотеки scikit-learn.
    """
    
    def __init__(self, model_class: str, params: Dict[str, Any]):
        """
        Инициализирует sklearn-совместимую модель.

        Args:
            model_class (str): Полный путь к классу модели в scikit-learn.
                               Например, 'sklearn.linear_model.LogisticRegression'.
            params (Dict[str, Any]): Словарь с параметрами для конструктора модели.
        """
        self.model_class_path = model_class
        self.params = params
        
        try:
            # Динамически импортируем и создаем класс модели
            module_path, class_name = model_class.rsplit('.', 1)
            module = __import__(module_path, fromlist=[class_name])
            model_constructor = getattr(module, class_name)
            self.model: BaseEstimator = model_constructor(**self.params)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Не удалось импортировать или создать класс модели: {model_class}") from e

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> None:
        """Обучает модель. `kwargs` игнорируются для совместимости."""
        print(f"Обучение модели {self.model.__class__.__name__}...")
        self.model.fit(X_train, y_train)

    def predict(self, X: pd.DataFrame) -> Any:
        """Возвращает предсказания классов или значения."""
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> Any:
        """Возвращает предсказания вероятностей или значения."""
        if hasattr(self.model, 'predict_proba'):
            # Для классификаторов возвращаем вероятности класса "1"
            return self.model.predict_proba(X)[:, 1]
        else:
            # Для регрессоров возвращаем просто предсказание
            return self.model.predict(X)

    def save(self, filepath: str) -> None:
        """Сохраняет модель с помощью joblib."""
        print(f"Сохранение модели в {filepath}")
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'SklearnModel':
        """Загружает модель с помощью joblib."""
        print(f"Загрузка модели из {filepath}")
        return joblib.load(filepath)