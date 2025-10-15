# src/models/lgbm.py

from typing import Any, Dict

import joblib
import lightgbm as lgb
import pandas as pd

from .base import ModelInterface # Импортируем наш базовый "контракт"

# ==================================================================================
# LGBMModel
# ==================================================================================
class LGBMModel(ModelInterface):
    """
    Класс-обертка для LightGBM Classifier / Regressor.
    """
    
    def __init__(self, params: Dict[str, Any]):
        """
        Инициализирует модель LightGBM.

        Args:
            params (Dict[str, Any]): Словарь с параметрами для модели.
                                     Например, {'objective': 'binary', ...}
        """
        self.params = params
        
        # Выбираем класс в зависимости от задачи (классификация или регрессия)
        objective = self.params.get('objective', '').lower()
        
        if 'regression' in objective or 'mae' in objective or 'mse' in objective:
            self.is_regressor = True
            self.model = lgb.LGBMRegressor(**self.params)
        else:
            self.is_regressor = False
            self.model = lgb.LGBMClassifier(**self.params)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> None:
        """
        Обучает модель LightGBM.

        Принимает `eval_set` и другие параметры для `fit` напрямую,
        что позволяет использовать `early_stopping_rounds`.
        """
        print("Обучение модели LightGBM...")
        self.model.fit(X_train, y_train, **kwargs)

    def predict(self, X: pd.DataFrame) -> Any:
        """Возвращает предсказания классов (для классификации) или значения (для регрессии)."""
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> Any:
        """
        Возвращает предсказания вероятностей (для классификации) или
        числовые предсказания (для регрессии).
        """
        if self.is_regressor:
            # Для регрессора predict_proba не существует, возвращаем обычные предсказания
            return self.model.predict(X)
        else:
            # Для классификации возвращаем только вероятности для класса "1"
            return self.model.predict_proba(X)[:, 1]

    def save(self, filepath: str) -> None:
        """Сохраняет модель с помощью joblib."""
        print(f"Сохранение модели в {filepath}")
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'LGBMModel':
        """Загружает модель с помощью joblib."""
        print(f"Загрузка модели из {filepath}")
        return joblib.load(filepath)