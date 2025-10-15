# src/features/aggregation.py

import pandas as pd
from typing import List, Dict, Any

from .base import FeatureGenerator, FitStrategy

# =a================================================================================
# AggregationGenerator (Стандартные групповые статистики)
# ==================================================================================
class AggregationGenerator(FeatureGenerator):
    """
    Создает агрегированные признаки с помощью операций `groupby().agg()`.

    Суммирует поведение сущностей по всей их истории. Поддерживает расширенный
    список статистических функций.

    Параметры:
        name (str): Уникальное имя для шага.
        group_keys (List[str]): Список колонок для группировки.
        group_values (List[str]): Список колонок для агрегации.
        agg_funcs (List[str]): Список функций агрегации.
            Поддерживаются: 'mean', 'std', 'sum', 'median', 'min', 'max', 'count',
            'nunique', 'skew', 'kurt' (для kurtosis).
    """
    fit_strategy: FitStrategy = "combined"

    def __init__(self, name: str, group_keys: List[str], group_values: List[str], agg_funcs: List[str]):
        super().__init__(name)
        self.group_keys = group_keys
        self.group_values = group_values
        self.agg_funcs = agg_funcs
        self.agg_df_: pd.DataFrame = None

    def fit(self, data: pd.DataFrame) -> None:
        """
        Вычисляет и сохраняет агрегированные статистики. Для стабильности
        признаков этот метод часто вызывается на объединенных данных (train + test).
        """
        print(f"[{self.name}] Обучение AggregationGenerator: группировка по {self.group_keys}")
        
        agg_df = data.groupby(self.group_keys)[self.group_values].agg(self.agg_funcs)
        
        new_cols = [f"{'_'.join(self.group_keys)}_{col[1]}_{col[0]}" for col in agg_df.columns.values]
        agg_df.columns = new_cols
        agg_df.reset_index(inplace=True)
        
        self.agg_df_ = agg_df
        print(f"  - Создано {len(self.agg_df_)} агрегированных записей с {self.agg_df_.shape[1]} признаками.")

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Присоединяет вычисленные агрегации к исходному датафрейму."""
        print(f"[{self.name}] Применение AggregationGenerator к {len(data)} строкам.")
        df = pd.merge(data, self.agg_df_, on=self.group_keys, how='left')
        return df

# ==================================================================================
# RollingAggregationGenerator (Статистики в скользящем временном окне)
# ==================================================================================
class RollingAggregationGenerator(FeatureGenerator):
    """
    Создает агрегированные признаки в скользящем временном окне для каждой группы.

    Вычисляет статистики (например, среднее, сумму) для заданных признаков
    за предыдущий период времени (например, за последние 7 дней).

    ВАЖНО: Результат сдвигается на 1 шаг (lagged), чтобы избежать утечки
    данных из текущего события. Признак описывает состояние *до* текущего момента.

    Параметры:
        name (str): Уникальное имя для шага.
        group_keys (List[str]): Список колонок для группировки (например, ['user_id']).
        date_col (str): Колонка с датой/временем, по которой будет строиться окно.
        value_cols (List[str]): Список числовых колонок для агрегации в окне.
        window_sizes (List[str]): Список размеров окон в формате pandas
            (например, '3D', '7D', '30D' - 3 дня, 7 дней, 30 дней).
        agg_funcs (List[str]): Список функций агрегации ('mean', 'sum', 'count').
    """

    fit_strategy: FitStrategy = "combined"

    def __init__(self, name: str, group_keys: List[str], date_col: str, 
                 value_cols: List[str], window_sizes: List[str], agg_funcs: List[str]):
        super().__init__(name)
        self.group_keys = group_keys
        self.date_col = date_col
        self.value_cols = value_cols
        self.window_sizes = window_sizes
        self.agg_funcs = agg_funcs

    def fit(self, data: pd.DataFrame) -> None:
        """Это stateless преобразование, обучение не требуется."""
        print(f"[{self.name}] RollingAggregationGenerator не требует обучения.")
        pass

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Вычисляет и присоединяет признаки в скользящем окне."""
        df = data.copy()
        print(f"[{self.name}] Применение RollingAggregationGenerator к {len(df)} строкам.")

        # Убедимся, что колонка с датой имеет правильный тип
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        
        # Сортировка - обязательное условие для корректной работы rolling
        df.sort_values(by=self.group_keys + [self.date_col], inplace=True)
        
        grouped = df.groupby(self.group_keys)
        
        for value_col in self.value_cols:
            for window in self.window_sizes:
                for func in self.agg_funcs:
                    new_col_name = f"{'_'.join(self.group_keys)}_{value_col}_rolling_{window}_{func}"
                    print(f"  - Создание признака: {new_col_name}")
                    
                    # Вычисляем rolling агрегацию
                    rolling_agg = grouped[value_col].rolling(window, on=self.date_col).agg(func)
                    
                    # Сдвигаем результат на 1, чтобы избежать утечки.
                    # reset_index нужен, чтобы вернуть group_keys для правильного merge
                    lagged_agg = rolling_agg.reset_index(0, drop=True).shift(1)
                    
                    df[new_col_name] = lagged_agg

        return df