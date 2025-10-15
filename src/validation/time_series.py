# src/validation/time_series.py

from typing import Iterator, Tuple, Optional
import pandas as pd
import numpy as np

# Импортируем интерфейс из соседнего файла base.py
from .base import BaseSplitter

# ==================================================================================
# DateCutoffSplitter
# ==================================================================================
class DateCutoffSplitter(BaseSplitter):
    """
    Реализует простую hold-out валидацию по временной отсечке.
    Создает один-единственный фолд, где train - все данные до cutoff_day,
    а valid - все данные после.
    """
    def __init__(self, date_col: str, cutoff_day: int):
        self.date_col = date_col
        self.cutoff_day = cutoff_day

    def split(self, data: pd.DataFrame, y: Optional[pd.Series] = None, groups: Optional[pd.Series] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        if self.date_col not in data.columns:
            raise ValueError(f"Колонка с датой '{self.date_col}' не найдена.")

        dates = pd.to_numeric(data[self.date_col])
        train_indices = np.where(dates <= self.cutoff_day)[0]
        valid_indices = np.where(dates > self.cutoff_day)[0]

        print(f"--- [DateCutoffSplitter] ---")
        print(f"  Train: дни <= {self.cutoff_day} (размер: {len(train_indices)})")
        print(f"  Valid: дни > {self.cutoff_day} (размер: {len(valid_indices)})")
        
        yield train_indices, valid_indices

    def get_n_splits(self, *args, **kwargs) -> int:
        return 1

# ==================================================================================
# RollingTimeCV
# ==================================================================================
class RollingTimeCV(BaseSplitter):
    """
    Создает фолды для кросс-валидации на временных данных, "прокатывая"
    окно валидации назад во времени.

    Параметры:
        date_col (str): Название колонки с датой или номером дня.
        n_splits (int): Количество фолдов для создания.
        valid_duration (int): Длительность валидационного периода.
        gap_duration (int): Длительность "зазора" между train и valid.
    """
    def __init__(self, date_col: str, n_splits: int = 5, valid_duration: int = 7, gap_duration: int = 0):
        if n_splits <= 0 or valid_duration <= 0 or gap_duration < 0:
            raise ValueError("n_splits и valid_duration должны быть > 0, gap_duration >= 0.")
        
        self.date_col = date_col
        self.n_splits = n_splits
        self.valid_duration = valid_duration
        self.gap_duration = gap_duration

    def split(self, data: pd.DataFrame, y: Optional[pd.Series] = None, groups: Optional[pd.Series] = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        if self.date_col not in data.columns:
            raise ValueError(f"Колонка с датой '{self.date_col}' не найдена в DataFrame.")

        dates = pd.to_numeric(data[self.date_col])
        min_date, max_date = dates.min(), dates.max()
        cycle_duration = self.valid_duration + self.gap_duration

        print("--- [RollingTimeCV] ---")
        print(f"Диапазон дней: [{min_date}, {max_date}], Длительность Valid: {self.valid_duration}, Зазор: {self.gap_duration}")
        print("-----------------------------------------")

        for i in range(self.n_splits):
            shift = i * cycle_duration
            valid_end_date = max_date - shift
            valid_start_date = valid_end_date - self.valid_duration
            train_end_date = valid_start_date - self.gap_duration

            if train_end_date < min_date:
                print(f"ПРЕДУПРЕЖДЕНИЕ: Недостаточно данных для создания фолда {i + 1}. Остановка.")
                break

            train_mask = (dates <= train_end_date)
            valid_mask = (dates > valid_start_date) & (dates <= valid_end_date)
            train_indices = np.where(train_mask)[0]
            valid_indices = np.where(valid_mask)[0]

            if len(train_indices) == 0 or len(valid_indices) == 0:
                print(f"ПРЕДУПРЕЖДЕНИЕ: Фолд {i + 1} пуст. Пропуск.")
                continue

            print(f"Фолд {i + 1}/{self.n_splits}:")
            print(f"  Train: дни <= {train_end_date} (размер: {len(train_indices)})")
            print(f"  Valid: дни ({valid_start_date}, {valid_end_date}] (размер: {len(valid_indices)})")
            
            yield train_indices, valid_indices

    def get_n_splits(self, *args, **kwargs) -> int:
        return self.n_splits