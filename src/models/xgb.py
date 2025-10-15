# src/models/xgb.py

from typing import Any, Dict

import joblib
import pandas as pd
import xgboost as xgb

from .base import ModelInterface # Импортируем наш базовый "контракт"

# ==================================================================================
# XGBModel
# ==================================================================================
class XGBModel(ModelInterface):
    """
    Класс-обертка для XGBoost Classifier / Regressor.
    """
    
    def __init__(self, params: Dict[str, Any]):
        """
        Инициализирует модель XGBoost.

        Args:
            params (Dict[str, Any]): Словарь с параметрами для модели.
                                     Например, {'objective': 'binary:logistic', ...}
        """
        self.params = params
        # Выбираем класс в зависимости от задачи (классификация или регрессия)
        if 'regressor' in str(self.params.get('objective', '')).lower():
            self.model = xgb.XGBRegressor(**self.params)
        else:
            self.model = xgb.XGBClassifier(**self.params)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> None:
        """
        Обучает модель XGBoost.

        Принимает `eval_set` и другие параметры для `fit` напрямую.
        """
        print("Обучение модели XGBoost...")
        self.model.fit(X_train, y_train, **kwargs)

    def predict(self, X: pd.DataFrame) -> Any:
        """Возвращает предсказания классов (для классификации)."""
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> Any:
        """
        Возвращает предсказания вероятностей (для классификации) или
        числовые предсказания (для регрессии).
        """
        if hasattr(self.model, 'predict_proba'):
            # Для классификации возвращаем только вероятности для класса "1"
            return self.model.predict_proba(X)[:, 1]
        else:
            # Для регрессии predict_proba не существует, возвращаем обычные предсказания
            return self.model.predict(X)

    def save(self, filepath: str) -> None:
        """Сохраняет модель с помощью joblib."""
        print(f"Сохранение модели в {filepath}")
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'XGBModel':
        """Загружает модель с помощью joblib."""
        print(f"Загрузка модели из {filepath}")
        return joblib.load(filepath)