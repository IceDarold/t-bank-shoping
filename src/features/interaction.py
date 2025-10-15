# src/features/interaction.py

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
from itertools import combinations

from .base import FeatureGenerator

# ==================================================================================
# NumericalInteractionGenerator
# ==================================================================================
class NumericalInteractionGenerator(FeatureGenerator):
    """
    Создает взаимодействия между парами числовых признаков.

    Для каждой пары колонок из списка `cols` применяются заданные
    математические операции.

    Параметры:
        name (str): Уникальное имя для шага.
        cols (List[str]): Список числовых колонок для создания взаимодействий.
        operations (List[str]): Список операций для применения.
            Доступные операции: 'add', 'subtract', 'multiply', 'divide'.
    """
    def __init__(self, name: str, cols: List[str], operations: List[str] = ['add', 'subtract', 'multiply', 'divide']):
        super().__init__(name)
        if len(cols) < 2:
            raise ValueError("Для создания взаимодействий требуется как минимум 2 колонки.")
        self.cols = cols
        self.operations = operations
        self.epsilon = 1e-6 # Для безопасного деления

    def fit(self, data: pd.DataFrame) -> None:
        """Это stateless преобразование, обучение не требуется."""
        for col in self.cols:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data")
        logging.info(f"[{self.name}] NumericalInteractionGenerator не требует обучения.")
        pass

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Создает новые признаки-взаимодействия."""
        for col in self.cols:
            if col not in data.columns:
                raise ValueError(f"Column '{col}' not found in data")
        df = data
        logging.info(f"[{self.name}] Создание числовых взаимодействий для колонок: {self.cols}")

        # Генерируем все уникальные пары колонок
        for c1, c2 in combinations(self.cols, 2):
            if 'add' in self.operations:
                df[f"{c1}_add_{c2}"] = df[c1] + df[c2]
            if 'subtract' in self.operations:
                df[f"{c1}_sub_{c2}"] = df[c1] - df[c2]
                df[f"{c2}_sub_{c1}"] = df[c2] - df[c1] # Разность несимметрична
            if 'multiply' in self.operations:
                df[f"{c1}_mul_{c2}"] = df[c1] * df[c2]
            if 'divide' in self.operations:
                try:
                    df[f"{c1}_div_{c2}"] = df[c1] / (df[c2] + self.epsilon)
                    df[f"{c2}_div_{c1}"] = df[c2] / (df[c1] + self.epsilon) # Деление несимметрично
                except Exception as e:
                    logging.error(f"Error in division operations for {c1} and {c2}: {e}")
                    raise

        return df

# ==================================================================================
# CategoricalInteractionGenerator
# ==================================================================================
class CategoricalInteractionGenerator(FeatureGenerator):
    """
    Создает взаимодействия между категориальными признаками путем их конкатенации.

    Например, `city='Moscow'` и `device='Desktop'` превратятся в `city_device='Moscow_Desktop'`.
    Это позволяет модели улавливать зависимости, специфичные для комбинации категорий.

    Параметры:
        name (str): Уникальное имя для шага.
        cols (List[List[str]]): Список списков колонок. Взаимодействия будут
            созданы для каждой группы колонок во внутреннем списке.
            Пример: [['city', 'device'], ['product_brand', 'country']]
    """
    def __init__(self, name: str, cols: List[List[str]]):
        super().__init__(name)
        self.cols_groups = cols

    def fit(self, data: pd.DataFrame) -> None:
        """Это stateless преобразование, обучение не требуется."""
        for group in self.cols_groups:
            for col in group:
                if col not in data.columns:
                    raise ValueError(f"Column '{col}' not found in data")
        logging.info(f"[{self.name}] CategoricalInteractionGenerator не требует обучения.")
        pass

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Создает новые конкатенированные категориальные признаки."""
        for group in self.cols_groups:
            for col in group:
                if col not in data.columns:
                    raise ValueError(f"Column '{col}' not found in data")
        df = data
        logging.info(f"[{self.name}] Создание категориальных взаимодействий.")

        for group in self.cols_groups:
            new_col_name = "_".join(group)
            # Убедимся, что все колонки строкового типа перед конкатенацией
            try:
                df[new_col_name] = df[group].astype(str).agg('_'.join, axis=1)
                logging.info(f"  - Создана колонка: {new_col_name}")
            except Exception as e:
                logging.error(f"Error creating column {new_col_name}: {e}")
                raise

        return df
        
# ==================================================================================
# NumCatInteractionGenerator
# ==================================================================================
class NumCatInteractionGenerator(FeatureGenerator):
    """
    Создает взаимодействия между числовыми и категориальными признаками.

    Вычисляет отклонение числового признака от среднего значения этого признака
    внутри его категории. Например, `income_deviation_from_city_mean`.
    Это очень мощный признак, который показывает, насколько значение является
    "типичным" для своей группы.

    Параметры:
        name (str): Уникальное имя для шага.
        interactions (Dict[str, List[str]]): Словарь, где ключ - это
            категориальный признак (группа), а значение - список числовых
            признаков, для которых нужно посчитать отклонение.
    """
    def __init__(self, name: str, interactions: Dict[str, List[str]]):
        super().__init__(name)
        self.interactions = interactions
        self.group_means_: Dict[str, pd.Series] = {}
        self.overall_means_: Dict[str, float] = {}
        self.epsilon = 1e-6

    def fit(self, data: pd.DataFrame) -> None:
        """
        Вычисляет и сохраняет средние значения для каждой категории
        ТОЛЬКО на обучающих данных.
        """
        for cat_col, num_cols in self.interactions.items():
            if cat_col not in data.columns:
                raise ValueError(f"Column '{cat_col}' not found in data")
            for num_col in num_cols:
                if num_col not in data.columns:
                    raise ValueError(f"Column '{num_col}' not found in data")
        logging.info(f"[{self.name}] Обучение NumCatInteractionGenerator.")
        for cat_col, num_cols in self.interactions.items():
            for num_col in num_cols:
                try:
                    group_means = data.groupby(cat_col)[num_col].mean()
                    self.group_means_[f"{num_col}_in_{cat_col}"] = group_means
                    logging.info(f"  - Вычислены средние для '{num_col}' по группам '{cat_col}'.")
                except Exception as e:
                    logging.error(f"Error computing group means for {num_col} in {cat_col}: {e}")
                    raise
        # Precompute overall means to avoid data leakage
        for num_cols in self.interactions.values():
            for num_col in num_cols:
                if num_col not in self.overall_means_:
                    self.overall_means_[num_col] = data[num_col].mean()
                
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Вычисляет и добавляет признаки отклонения от среднего по группе.
        """
        for cat_col, num_cols in self.interactions.items():
            if cat_col not in data.columns:
                raise ValueError(f"Column '{cat_col}' not found in data")
            for num_col in num_cols:
                if num_col not in data.columns:
                    raise ValueError(f"Column '{num_col}' not found in data")
        df = data
        logging.info(f"[{self.name}] Применение NumCatInteractionGenerator к {len(df)} строкам.")
        for cat_col, num_cols in self.interactions.items():
            for num_col in num_cols:
                # 1. Присоединяем средние по группе к датафрейму
                group_means = self.group_means_[f"{num_col}_in_{cat_col}"]
                try:
                    df_merged = df[[cat_col]].merge(group_means.rename('group_mean'),
                                                    left_on=cat_col, right_index=True, how='left')
                except Exception as e:
                    logging.error(f"Error merging group means for {num_col} in {cat_col}: {e}")
                    raise

                # 2. Заполняем пропуски (для категорий, которых не было в трейне)
                #    общим средним по числовой колонке.
                df_merged['group_mean'] = df_merged['group_mean'].fillna(self.overall_means_[num_col])

                # 3. Вычисляем и создаем новые признаки
                try:
                    df[f"{num_col}_div_by_{cat_col}_mean"] = df[num_col] / (df_merged['group_mean'] + self.epsilon)
                    df[f"{num_col}_sub_by_{cat_col}_mean"] = df[num_col] - df_merged['group_mean']
                except Exception as e:
                    logging.error(f"Error computing interaction features for {num_col} in {cat_col}: {e}")
                    raise

        return df