# src/stack.py

import warnings
from pathlib import Path
import time
from typing import List

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf, ListConfig
import wandb

# Импортируем наши собственные модули
from src import utils
from src.models.base import ModelInterface
from src.metrics.base import MetricInterface

warnings.filterwarnings("ignore")


def create_meta_dataset(configs: ListConfig, data_path: Path, id_col: str, pred_col_name: str) -> pd.DataFrame:
    """
    Универсальная функция для загрузки и объединения предсказаний базовых моделей.
    """
    dfs = []
    print(f"\n--- Сборка мета-датасета (источник: '{pred_col_name}') ---")
    
    for model_cfg in configs:
        # В `train_using` ключ `oof_path`, в `predict_using` - `path`
        path_key = 'oof_path' if 'oof_path' in model_cfg else 'path'
        path = data_path / model_cfg[path_key]
        
        if not path.exists():
            raise FileNotFoundError(f"Файл с предсказаниями не найден: {path}")
        
        df = pd.read_csv(path)
        # Переименовываем колонку с предсказаниями в `pred_model-name`
        df.rename(columns={pred_col_name: f"pred_{model_cfg.name}"}, inplace=True)
        dfs.append(df)
        print(f"  - Загружены предсказания от: {model_cfg.name}")
        
    # Последовательно объединяем все датафреймы по ID
    meta_df = dfs[0]
    for i in range(1, len(dfs)):
        meta_df = pd.merge(meta_df, dfs[i], on=id_col, how='left')
        
    return meta_df


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def stack(cfg: DictConfig) -> None:
    """
    Главный пайплайн для стекинга.
    1. Собирает обучающий мета-датасет из OOF-предсказаний.
    2. Собирает тестовый мета-датасет из предсказаний на тесте.
    3. Обучает мета-модель и делает финальное предсказание.
    """
    start_time = time.time()
    stack_cfg = cfg.stacking
    
    # === 1. Инициализация W&B и подготовка ===
    utils.seed_everything(cfg.globals.seed)
    output_dir = Path.cwd()
    
    run_name = f"stacking-{stack_cfg.name}-{utils.get_timestamp()}"
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=run_name,
        job_type="stacking",
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=["stacking"],
    )
    
    print("--- Запуск пайплайна стекинга ---")
    print(OmegaConf.to_yaml(stack_cfg))
    
    data_path = Path(hydra.utils.get_original_cwd()) / "data"
    
    # --- 2. Создание ОБУЧАЮЩЕГО мета-датасета (на OOF) ---
    meta_train_df = create_meta_dataset(
        configs=stack_cfg.train_using,
        data_path=data_path,
        id_col=cfg.globals.id_col,
        pred_col_name='oof_preds' # OOF файлы имеют колонку 'oof_preds'
    )
    # Добавляем исходный таргет
    raw_train_df = pd.read_csv(data_path / cfg.data.processed_path / cfg.data.train_file)
    meta_train_df = pd.merge(meta_train_df, raw_train_df[[cfg.globals.id_col, cfg.globals.target_col]], on=cfg.globals.id_col, how='left')
    
    feature_cols = [col for col in meta_train_df.columns if col.startswith('pred_')]
    X_train = meta_train_df[feature_cols]
    y_train = meta_train_df[cfg.globals.target_col]

    # --- 3. Создание ТЕСТОВОГО мета-датасета (для инференса) ---
    meta_test_df = create_meta_dataset(
        configs=stack_cfg.predict_using,
        data_path=data_path,
        id_col=cfg.globals.id_col,
        pred_col_name=cfg.globals.target_col # Файлы сабмишенов имеют колонку с именем таргета
    )
    X_test = meta_test_df[feature_cols]
    
    print(f"\nСозданы мета-датасеты с {len(feature_cols)} признаками.")

    # --- 4. Оценка качества стекинга на CV (опционально, но рекомендуется) ---
    cv_splitter = hydra.utils.instantiate(cfg.validation.strategy)
    main_metric: MetricInterface = hydra.utils.instantiate(cfg.metric.main)
    fold_scores = []
    print("\n--- Оценка качества стекинга на кросс-валидации (на OOF-данных) ---")
    for fold, (train_idx, valid_idx) in enumerate(cv_splitter.split(X_train, y_train)):
        X_train_fold, y_train_fold = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_valid_fold, y_valid_fold = X_train.iloc[valid_idx], y_train.iloc[valid_idx]
        
        cv_meta_model: ModelInterface = hydra.utils.instantiate(stack_cfg.meta_model)
        cv_meta_model.fit(X_train_fold, y_train_fold)
        valid_preds = cv_meta_model.predict_proba(X_valid_fold)
        score = main_metric(y_valid_fold.values, valid_preds)
        fold_scores.append(score)
        print(f"  - Скор на фолде {fold + 1}: {score:.5f}")

    oof_score_mean = np.mean(fold_scores)
    run.summary["stack_cv_score"] = oof_score_mean
    print(f"Итоговый CV-скор стекинга: {oof_score_mean:.5f}")

    # --- 5. Обучение финальной мета-модели и инференс ---
    print("\n--- Обучение финальной мета-модели на всех OOF-данных и инференс ---")
    final_meta_model: ModelInterface = hydra.utils.instantiate(stack_cfg.meta_model)
    final_meta_model.fit(X_train, y_train)
    
    final_predictions = final_meta_model.predict_proba(X_test)
    
    # --- 6. Сохранение финального сабмишена ---
    submission_path = data_path / cfg.data.submissions_path
    submission_filepath = submission_path / f"submission_stack_{stack_cfg.name}.csv"
    
    submission_df = pd.DataFrame({cfg.globals.id_col: meta_test_df[cfg.globals.id_col], cfg.globals.target_col: final_predictions})
    submission_df.to_csv(submission_filepath, index=False)
    
    run.log_artifact(wandb.Artifact(name=f"submission-{stack_cfg.name}", type="submission").add_file(str(submission_filepath)))
    print(f"\nФинальный сабмишен сохранен в: {submission_filepath}")
    
    end_time = time.time()
    print(f"Пайплайн стекинга завершен за {end_time - start_time:.2f} секунд.")
    run.finish()

if __name__ == "__main__":
    try:
        stack()
    except Exception as e:
        print(f"\nКритическая ошибка во время выполнения: {e}")
        if wandb.run:
            print("Завершение W&B run с ошибкой...")
            wandb.finish(exit_code=1)
        raise