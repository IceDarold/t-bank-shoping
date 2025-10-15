# src/features/datetime.py

import numpy as np
import pandas as pd
from typing import List, Dict, Any

from .base import FeatureGenerator

# ==================================================================================
# DatetimeFeatureGenerator
# ==================================================================================
class DatetimeFeatureGenerator(FeatureGenerator):
    """
    Извлекает различные компоненты из колонок с датой и временем.

    Преобразует колонки с датой/временем в набор числовых и циклических
    признаков, которые могут быть использованы моделями.

    Параметры:
        name (str): Уникальное имя для шага.
        cols (List[str]): Список колонок для обработки.
        components (List[str]): Список компонентов для извлечения.
            Доступные компоненты:
            - 'year', 'month', 'day', 'hour', 'minute', 'second'
            - 'weekday' (день недели, 0=Пн), 'dayofyear', 'weekofyear'
            - 'quarter'
            - 'is_weekend', 'is_month_start', 'is_month_end'
            - 'time_of_day' ('Morning', 'Afternoon', 'Evening', 'Night')
            - 'cyclical' (создает sin/cos преобразования для month, weekday, hour)
    """
    def __init__(self, name: str, cols: List[str], components: List[str]):
        super().__init__(name)
        self.cols = cols
        self.components = set(components)

    def fit(self, data: pd.DataFrame) -> None:
        """Это stateless преобразование, обучение не требуется."""
        print(f"[{self.name}] DatetimeFeatureGenerator не требует обучения.")
        pass

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Создает новые признаки на основе даты и времени."""
        df = data.copy()
        print(f"[{self.name}] Извлечение компонентов даты/времени из: {self.cols}")

        for col in self.cols:
            # Преобразуем колонку в datetime, если она еще не в этом формате
            dt_series = pd.to_datetime(df[col], errors='coerce')

            if 'year' in self.components: df[f'{col}_year'] = dt_series.dt.year
            if 'month' in self.components: df[f'{col}_month'] = dt_series.dt.month
            if 'day' in self.components: df[f'{col}_day'] = dt_series.dt.day
            if 'hour' in self.components: df[f'{col}_hour'] = dt_series.dt.hour
            if 'minute' in self.components: df[f'{col}_minute'] = dt_series.dt.minute
            if 'second' in self.components: df[f'{col}_second'] = dt_series.dt.second
            if 'weekday' in self.components: df[f'{col}_weekday'] = dt_series.dt.weekday
            if 'dayofyear' in self.components: df[f'{col}_dayofyear'] = dt_series.dt.dayofyear
            if 'weekofyear' in self.components: df[f'{col}_weekofyear'] = dt_series.dt.isocalendar().week.astype(int)
            if 'quarter' in self.components: df[f'{col}_quarter'] = dt_series.dt.quarter

            if 'is_weekend' in self.components: df[f'{col}_is_weekend'] = (dt_series.dt.weekday >= 5).astype(int)
            if 'is_month_start' in self.components: df[f'{col}_is_month_start'] = dt_series.dt.is_month_start.astype(int)
            if 'is_month_end' in self.components: df[f'{col}_is_month_end'] = dt_series.dt.is_month_end.astype(int)
            
            if 'time_of_day' in self.components:
                hour = dt_series.dt.hour
                bins = [0, 6, 12, 18, 24]
                labels = ['Night', 'Morning', 'Afternoon', 'Evening']
                df[f'{col}_time_of_day'] = pd.cut(hour, bins=bins, labels=labels, right=False, ordered=False)

            if 'cyclical' in self.components:
                # Синус/косинус преобразования для захвата цикличности
                if 'month' in self.components:
                    df[f'{col}_month_sin'] = np.sin(2 * np.pi * dt_series.dt.month / 12)
                    df[f'{col}_month_cos'] = np.cos(2 * np.pi * dt_series.dt.month / 12)
                if 'weekday' in self.components:
                    df[f'{col}_weekday_sin'] = np.sin(2 * np.pi * dt_series.dt.weekday / 7)
                    df[f'{col}_weekday_cos'] = np.cos(2 * np.pi * dt_series.dt.weekday / 7)
                if 'hour' in self.components:
                    df[f'{col}_hour_sin'] = np.sin(2 * np.pi * dt_series.dt.hour / 24)
                    df[f'{col}_hour_cos'] = np.cos(2 * np.pi * dt_series.dt.hour / 24)

        return df

# ==================================================================================
# DateDifferenceGenerator
# ==================================================================================
class DateDifferenceGenerator(FeatureGenerator):
    """
    Вычисляет разницу между двумя колонками с датой/временем.

    Параметры:
        name (str): Уникальное имя для шага.
        col1 (str): Первая (более поздняя) колонка с датой.
        col2 (str): Вторая (более ранняя) колонка с датой.
        unit (str): Единица измерения для разницы ('D' для дней, 'h' для часов и т.д.).
    """
    def __init__(self, name: str, col1: str, col2: str, unit: str = 'D'):
        super().__init__(name)
        self.col1 = col1
        self.col2 = col2
        self.unit = unit

    def fit(self, data: pd.DataFrame) -> None:
        """Это stateless преобразование, обучение не требуется."""
        print(f"[{self.name}] DateDifferenceGenerator не требует обучения.")
        pass

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Создает новый признак с разницей во времени."""
        df = data.copy()
        print(f"[{self.name}] Вычисление разницы между {self.col1} и {self.col2}")
        
        dt1 = pd.to_datetime(df[self.col1], errors='coerce')
        dt2 = pd.to_datetime(df[self.col2], errors='coerce')
        
        time_delta = (dt1 - dt2).dt.total_seconds()
        
        # Конвертируем секунды в нужную единицу
        if self.unit == 'D':
            divisor = 86400 # 24 * 60 * 60
        elif self.unit == 'h':
            divisor = 3600 # 60 * 60
        elif self.unit == 'm':
            divisor = 60
        elif self.unit == 's':
            divisor = 1
        else:
            raise ValueError(f"Неподдерживаемая единица измерения: {self.unit}")
            
        df[f"{self.col1}_minus_{self.col2}_in_{self.unit}"] = time_delta / divisor
        return df