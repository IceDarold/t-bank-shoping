# src/train.py

import warnings
from pathlib import Path
import time
from typing import List

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf
import wandb

# Импортируем наши собственные модули
from src import utils
from src.models.base import ModelInterface
from src.metrics.base import MetricInterface
from src.validation.base import BaseSplitter # Наш новый интерфейс для валидации

warnings.filterwarnings("ignore")


@hydra.main(config_path="conf", config_name="config", version_base=None)
def train(cfg: DictConfig) -> float:
    """
    Главный пайплайн для обучения модели.

    1. Инициализирует W&B run.
    2. Скачивает версионированный набор признаков из W&B Artifacts.
    3. Выполняет обучение в одном из двух режимов (CV или Full Data), используя
       гибкий модуль валидации.
    4. Сохраняет и логирует артефакты (модели, OOF-предсказания, сабмишен).
    """
    start_time = time.time()
    
    # === 1. Инициализация W&B и подготовка ===
    utils.seed_everything(cfg.globals.seed)
    # Hydra создает уникальную директорию для каждого запуска.
    # Мы будем использовать ее для временного хранения артефактов.
    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    
    run_name = f"{cfg.model._target_.split('.')[-1].replace('Model', '')}"
    run_name += f"-{cfg.feature_engineering.name}"
    run_name += "-FULL" if cfg.training.full_data else "-CV"
    
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        tags=cfg.wandb.tags,
        job_type="training",
    )
    
    print("--- Конфигурация эксперимента ---")
    print(OmegaConf.to_yaml(cfg))
    print("-----------------------------------")
    
    # === 2. Загрузка данных из артефакта W&B ===
    print("\n--- Загрузка набора признаков из артефакта W&B ---")
    
    feature_artifact_name = cfg.feature_engineering.name
    # Формируем полное имя артефакта для скачивания
    artifact_to_use = f"{cfg.wandb.entity}/{cfg.wandb.project}/{feature_artifact_name}:latest"
    print(f"Используется артефакт: {artifact_to_use}")
    
    # Эта команда скачивает артефакт и регистрирует его как входные данные для этого run'а
    try:
        artifact = run.use_artifact(artifact_to_use)
    except wandb.errors.CommError as e:
        print(f"\n[ОШИБКА] Не удалось найти артефакт '{artifact_to_use}'.")
        print("Убедитесь, что вы сначала запустили `make_features.py` с соответствующим конфигом.")
        raise e

    # Получаем путь к локальной папке, куда был скачан артефакт
    artifact_dir = Path(artifact.download())
    
    train_features_path = artifact_dir / f"train_{feature_artifact_name}.parquet"
    test_features_path = artifact_dir / f"test_{feature_artifact_name}.parquet"
    
    train_df = pd.read_parquet(train_features_path)
    test_df = pd.read_parquet(test_features_path)
    
    print("Признаки успешно загружены из артефакта W&B.")
    
    feature_cols = cfg.features.cols
    target_col = cfg.globals.target_col
    
    X = train_df[feature_cols]
    y = train_df[target_col]
    X_test = test_df[feature_cols]
    
    print(f"Используется {len(feature_cols)} признаков. train.shape={X.shape}, test.shape={X_test.shape}")
    
    # ==========================================================================
    # ❗️ ВЫБОР РЕЖИМА ОБУЧЕНИЯ
    # ==========================================================================
    if cfg.training.full_data:
        # --- РЕЖИМ 1: ОБУЧЕНИЕ НА ВСЕХ ДАННЫХ ---
        print("\n--- Режим: Обучение на 100% данных ---")
        
        model: ModelInterface = hydra.utils.instantiate(cfg.model)
        
        # Убираем параметры, специфичные для CV, если они есть
        fit_params = cfg.training.fit_params.copy()
        fit_params.pop('early_stopping_rounds', None)
        
        model.fit(X, y, **fit_params)
        test_preds = model.predict_proba(X_test)
        
        model_path = output_dir / "model_full_train.pkl"
        model.save(model_path)
        print(f"Модель, обученная на всех данных, сохранена в: {model_path}")

        oof_score_mean = -1.0
        
    else:
        # --- РЕЖИМ 2: ОБУЧЕНИЕ НА КРОСС-ВАЛИДАЦИИ ---
        print("\n--- Режим: Обучение на кросс-валидации ---")
        
        # Инстанциируем сплиттер из нашего нового модуля валидации
        splitter: BaseSplitter = hydra.utils.instantiate(cfg.validation)
        print(f"Стратегия валидации: {splitter.__class__.__name__} ({splitter.get_n_splits()} фолдов)")
        
        # Подготовка групп, если они требуются для сплиттера (например, GroupKFold)
        groups = None
        group_col = cfg.validation.get("group_col")
        if group_col:
            if group_col not in train_df.columns:
                raise ValueError(f"Колонка для группировки '{group_col}' не найдена в данных.")
            groups = train_df[group_col]
            print(f"Используется группировка по колонке: {group_col}")
        
        # Инициализация метрик
        main_metric: MetricInterface = hydra.utils.instantiate(cfg.metric.main)
        main_metric_name = main_metric.__class__.__name__.replace("Metric", "")
        
        additional_metrics: List[MetricInterface] = []
        if 'additional' in cfg.metric and cfg.metric.additional:
            for metric_cfg in cfg.metric.additional:
                metric_obj = hydra.utils.instantiate(metric_cfg)
                # Сохраняем имя метрики для логирования, как оно задано в конфиге
                metric_obj.name = metric_cfg.name 
                additional_metrics.append(metric_obj)

        oof_preds = np.zeros(len(train_df))
        test_preds = np.zeros(len(test_df))
        fold_scores = []
        
        # Получаем итератор для разделения данных
        split_iterator = splitter.split(data=train_df, y=y, groups=groups)
        
        for fold, (train_idx, valid_idx) in enumerate(split_iterator):
            fold_start_time = time.time()
            print(f"\n--- Фолд {fold + 1}/{splitter.get_n_splits()} ---")
            
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
            
            model: ModelInterface = hydra.utils.instantiate(cfg.model)
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_valid, y_valid)],
                **cfg.training.fit_params,
            )
            
            valid_preds_proba = model.predict_proba(X_valid)
            oof_preds[valid_idx] = valid_preds_proba
            test_preds += model.predict_proba(X_test) / splitter.get_n_splits()
            
            # Подготовка дополнительных данных для метрики (например, групп)
            metric_kwargs = {}
            if groups is not None:
                metric_kwargs['groups'] = groups.iloc[valid_idx]
            
            # Логирование
            log_dict = {"fold": fold + 1}
            fold_score = main_metric(y_valid.values, valid_preds_proba, **metric_kwargs)
            fold_scores.append(fold_score)
            log_dict[f"fold_score/{main_metric_name}"] = fold_score
            
            for metric_obj in additional_metrics:
                add_score = metric_obj(y_valid.values, valid_preds_proba, **metric_kwargs)
                log_dict[f"fold_score/{metric_obj.name}"] = add_score
            run.log(log_dict)
            
            # Сохранение модели
            model_path = output_dir / f"model_fold_{fold + 1}.pkl"
            model.save(model_path)
            
            fold_end_time = time.time()
            print(f"Скор на фолде {fold + 1} ({main_metric_name}): {fold_score:.5f} (за {fold_end_time - fold_start_time:.2f} с)")

        oof_score_mean = np.mean(fold_scores)
        oof_score_std = np.std(fold_scores)
        
        print(f"\n--- Итоговый результат CV ---")
        print(f"Средний OOF-скор ({main_metric_name}): {oof_score_mean:.5f} (Std: {oof_score_std:.5f})")
        
        run.summary[f"oof_score_mean"] = oof_score_mean
        run.summary[f"oof_score_std"] = oof_score_std

        # Сохранение OOF-предсказаний
        oof_df = pd.DataFrame({cfg.globals.id_col: train_df[cfg.globals.id_col], 'oof_preds': oof_preds})
        oof_path = output_dir / "oof_predictions.csv"
        oof_df.to_csv(oof_path, index=False)

    # === ФИНАЛЬНЫЙ ШАГ: СОХРАНЕНИЕ САБМИШЕНА И АРТЕФАКТОВ ===
    print("\n--- Сохранение артефактов ---")
    submission_df = pd.DataFrame({cfg.globals.id_col: test_df[cfg.globals.id_col], target_col: test_preds})
    submission_path = output_dir / "submission.csv"
    submission_df.to_csv(submission_path, index=False)
    
    output_artifact = wandb.Artifact(name=f"output-{run.id}", type="output")
    output_artifact.add_file(str(submission_path))
    if not cfg.training.full_data:
        output_artifact.add_file(str(oof_path))
    output_artifact.add_dir(str(output_dir), name="models")
    run.log_artifact(output_artifact)
    
    end_time = time.time()
    print(f"Все результаты сохранены в: {output_dir}")
    print(f"Пайплайн завершен за {end_time - start_time:.2f} секунд.")
    
    run.finish()
    
    return oof_score_mean


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"\nКритическая ошибка во время выполнения: {e}")
        if wandb.run:
            print("Завершение W&B run с ошибкой...")
            wandb.finish(exit_code=1)
        raise