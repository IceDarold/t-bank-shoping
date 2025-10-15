# src/models/catboost.py

from typing import Any, Dict, List

import joblib
import pandas as pd
import catboost as cb
from catboost.utils import get_gpu_device_count

from .base import ModelInterface # Импортируем наш базовый "контракт"

# ==================================================================================
# CatBoostModel
# ==================================================================================
class CatBoostModel(ModelInterface):
    """
    Класс-обертка для CatBoost Classifier / Regressor.
    """
    
    def __init__(self, params: Dict[str, Any]):
        """
        Инициализирует модель CatBoost.

        Args:
            params (Dict[str, Any]): Словарь с параметрами для модели.
                                     Например, {'iterations': 1000, ...}
        """
        self.params = params.copy() # Копируем, чтобы безопасно изменять
        
        # --- Автоматическая настройка параметров ---
        
        # 1. Настройка verbose по умолчанию для чистоты логов
        if 'verbose' not in self.params:
            self.params['verbose'] = False
            
        # 2. Автоматическое включение GPU, если он доступен и не указан task_type
        if 'task_type' not in self.params and get_gpu_device_count() > 0:
            print("Обнаружен GPU. Установка 'task_type': 'GPU' для CatBoost.")
            self.params['task_type'] = 'GPU'

        # 3. Выбираем класс в зависимости от задачи
        loss_function = self.params.get('loss_function', '').lower()
        regression_losses = {'rmse', 'mae', 'rmsle', 'quantile', 'mape'}
        
        if any(reg_loss in loss_function for reg_loss in regression_losses):
            self.is_regressor = True
            self.model = cb.CatBoostRegressor(**self.params)
        else:
            self.is_regressor = False
            self.model = cb.CatBoostClassifier(**self.params)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs: Any) -> None:
        """
        Обучает модель CatBoost.

        Автоматически находит категориальные признаки и передает их модели.
        Принимает `eval_set` и другие параметры для `fit` напрямую.
        """
        print("Обучение модели CatBoost...")
        
        # Находим категориальные признаки в данных
        cat_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_features:
            print(f"Найдены категориальные признаки для CatBoost: {cat_features}")
        
        # CatBoost предпочитает, чтобы категориальные признаки имели тип 'category'
        X_train_copy = X_train.copy()
        for col in cat_features:
            X_train_copy[col] = X_train_copy[col].astype('category')
        
        # Адаптируем eval_set, если он передан
        if 'eval_set' in kwargs:
            eval_set = kwargs['eval_set']
            X_valid, y_valid = eval_set[0]
            X_valid_copy = X_valid.copy()
            for col in cat_features:
                X_valid_copy[col] = X_valid_copy[col].astype('category')
            kwargs['eval_set'] = [(X_valid_copy, y_valid)]

        self.model.fit(X_train_copy, y_train, cat_features=cat_features, **kwargs)

    def predict(self, X: pd.DataFrame) -> Any:
        """Возвращает предсказания классов (для классификации) или значения (для регрессии)."""
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> Any:
        """
        Возвращает предсказания вероятностей (для классификации) или
        числовые предсказания (для регрессии).
        """
        if self.is_regressor:
            # Для регрессора predict_proba не существует
            return self.model.predict(X)
        else:
            # Для классификации возвращаем только вероятности для класса "1"
            return self.model.predict_proba(X)[:, 1]

    def save(self, filepath: str) -> None:
        """Сохраняет модель с помощью joblib."""
        print(f"Сохранение модели в {filepath}")
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> 'CatBoostModel':
        """Загружает модель с помощью joblib."""
        print(f"Загрузка модели из {filepath}")
        return joblib.load(filepath)