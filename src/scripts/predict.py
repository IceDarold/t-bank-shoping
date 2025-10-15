# src/predict.py

import glob
import os
import time
from pathlib import Path
from typing import Dict, Any

import hydra
import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from src.exceptions import ConfigurationError, DataLoadError, ModelLoadError, ValidationError

# Импортируем базовый класс, чтобы можно было загружать любую модель
from src.models.base import ModelInterface


def validate_config(cfg: DictConfig) -> None:
    """
    Validate configuration parameters.
    """
    if not cfg.inference.run_id:
        raise ConfigurationError("run_id must be provided.")
    if cfg.inference.id_col not in cfg.features.cols:
        raise ConfigurationError("ID column must be in feature columns.")
    if cfg.globals.target_col in cfg.features.cols:
        raise ConfigurationError("Target column must not be in feature columns.")


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def predict(cfg: DictConfig) -> None:
    """
    Скрипт для инференса (предсказаний).

    Загружает обученные модели из директории, соответствующей `run_id`,
    применяет их к указанному набору признаков и сохраняет файл сабмишена.
    """
    start_time = time.time()

    validate_config(cfg)

    print("--- Запуск инференса ---")
    
    # --- 1. Проверка и подготовка путей ---
    run_id = cfg.inference.run_id
    if not run_id:
        raise ConfigurationError("Необходимо указать ID запуска (`inference.run_id`) для инференса!")

    # Sanitize run_id to prevent path traversal
    run_id = os.path.basename(run_id)

    print(f"Используются модели из W&B run_id: {run_id}")

    # Ищем директорию с результатами по всему проекту
    # Это более надежно, чем жестко заданный путь.
    try:
        search_path = Path(hydra.utils.get_original_cwd()) / "outputs"
        models_path = next(search_path.glob(f"**/{run_id}"))
    except StopIteration:
        raise FileNotFoundError("Не удалось найти директорию для указанного run_id. "
                                "Проверьте, что такой запуск существует в папке 'outputs'.")

    print(f"Путь к моделям: {models_path}")
    
    # --- 2. Загрузка данных ---
    data_path = Path(hydra.utils.get_original_cwd()) / "data"
    features_path = data_path / cfg.data.features_path
    
    input_filename = cfg.inference.input_features_filename
    test_features_path = features_path / input_filename

    try:
        if not test_features_path.exists():
            raise FileNotFoundError("Файл с признаками не найден.")
        test_df = pd.read_parquet(test_features_path)
    except Exception as e:
        raise DataLoadError("Failed to load test features file.") from e
    
    # Загружаем список колонок из конфига, соответствующего эксперименту
    feature_cols = cfg.features.cols
    X_test = test_df[feature_cols]
    
    print(f"Загружены данные для инференса: {test_features_path.name} (shape: {X_test.shape})")

    # --- 3. Загрузка моделей и инференс ---
    # Сначала ищем модели, обученные на CV
    model_files = sorted(glob.glob(str(models_path / "model_fold_*.pkl")))
    
    # Если их нет, ищем одну модель, обученную на всех данных
    if not model_files:
        model_files = sorted(glob.glob(str(models_path / "model_full_train.pkl")))

    if not model_files:
        raise FileNotFoundError("В указанной директории не найдены обученные модели.")

    print(f"Найдено {len(model_files)} моделей для предсказания.")

    final_preds = None

    for model_path in tqdm(model_files, desc="Предсказание по моделям"):
        # Загружаем модель
        try:
            model: ModelInterface = ModelInterface.load(model_path)
        except Exception as e:
            raise ModelLoadError("Failed to load model file.") from e

        fold_preds = model.predict_proba(X_test)

        if final_preds is None:
            final_preds = fold_preds
        else:
            final_preds += fold_preds

    # Усредняем предсказания, если моделей было несколько
    if len(model_files) > 1:
        if len(model_files) > 0:
            final_preds /= len(model_files)
        else:
            raise ValueError("No models found for averaging predictions.")

    # --- 4. Сохранение сабмишена ---
    output_path = data_path / cfg.data.submissions_path
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Используем ID колонку из конфига инференса
    id_col = cfg.inference.id_col
    if id_col not in test_df.columns:
        raise ValueError(f"ID колонка '{id_col}', указанная в конфиге, не найдена в данных для инференса.")
        
    submission_df = pd.DataFrame({
        id_col: test_df[id_col],
        cfg.globals.target_col: final_preds
    })
    
    submission_filepath = output_path / f"submission_from_{run_id}.csv"
    submission_df.to_csv(submission_filepath, index=False)
    
    end_time = time.time()
    print("\n--- Инференс завершен ---")
    print(f"Файл сабмишена сохранен в: {submission_filepath}")
    print(f"Общее время выполнения: {end_time - start_time:.2f} секунд.")


if __name__ == "__main__":
    predict()