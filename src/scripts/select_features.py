# src/select_features.py

import time
from pathlib import Path

import hydra
import lightgbm as lgb
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import wandb

# Импортируем наши собственные модули
from src import utils


@hydra.main(config_path="conf", config_name="config", version_base=None)
def select_features(cfg: DictConfig) -> None:
    """
    Скрипт для отбора признаков (Feature Selection).

    1. Загружает набор признаков, указанный в `feature_engineering.name`.
    2. Обучает модель LightGBM на кросс-валидации.
    3. Агрегирует важность признаков (`feature_importance`) по всем фолдам.
    4. Сохраняет список `top_n` лучших признаков в локальный файл и как
       артефакт в Weights & Biases.
    """
    start_time = time.time()
    
    # --- 1. Инициализация W&B и подготовка ---
    run_name = f"select-features-{cfg.feature_engineering.name}-top{cfg.selection.top_n}"
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=run_name,
        job_type="feature-selection",
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    
    print("--- Запуск отбора признаков ---")
    print("Конфигурация отбора:")
    print(OmegaConf.to_yaml(cfg.selection))
    
    # --- 2. Загрузка данных из артефакта W&B ---
    print("\n--- Загрузка набора признаков из артефакта W&B ---")
    
    feature_artifact_name = cfg.feature_engineering.name
    artifact_to_use = f"{cfg.wandb.entity}/{cfg.wandb.project}/{feature_artifact_name}:latest"
    
    try:
        artifact = run.use_artifact(artifact_to_use)
    except wandb.errors.CommError as e:
        print(f"\n[ОШИБКА] Не удалось найти артефакт '{artifact_to_use}'.")
        raise e

    artifact_dir = Path(artifact.download())
    train_features_path = artifact_dir / f"train_{feature_artifact_name}.parquet"
    train_df = pd.read_parquet(train_features_path)
    
    print(f"Загружены признаки: {train_features_path.name} (shape: {train_df.shape})")

    # --- 3. Подготовка данных и CV ---
    target_col = cfg.globals.target_col
    # Исключаем ID и таргет, чтобы получить полный список исходных признаков
    all_feature_cols = [col for col in train_df.columns if col not in [cfg.globals.id_col, target_col]]
    
    X = train_df[all_feature_cols]
    y = train_df[target_col]
    
    cv_splitter = hydra.utils.instantiate(cfg.validation.strategy)
    
    # --- 4. Обучение на CV и сбор важности признаков ---
    feature_importances = pd.DataFrame(index=all_feature_cols)
    
    for fold, (train_idx, _) in tqdm(
        enumerate(cv_splitter.split(X, y)), 
        total=cv_splitter.get_n_splits(),
        desc="Оценка важности на фолдах"
    ):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        
        # Используем простую модель LGBM для скорости
        model = lgb.LGBMClassifier(**cfg.selection.model_params)
        model.fit(X_train, y_train)
        
        feature_importances[f'fold_{fold+1}'] = model.feature_importances_

    # --- 5. Агрегация и отбор лучших признаков ---
    feature_importances['mean'] = feature_importances.mean(axis=1)
    feature_importances.sort_values(by='mean', ascending=False, inplace=True)
    
    top_n = cfg.selection.top_n
    # Убедимся, что не пытаемся выбрать больше признаков, чем есть
    top_n = min(top_n, len(all_feature_cols))
    
    top_features = feature_importances.head(top_n).index.tolist()
    
    print(f"\n--- Топ 10 самых важных признаков (среднее по {cv_splitter.get_n_splits()} фолдам) ---")
    print(feature_importances['mean'].head(10))
    print("--------------------------------------")
    
    # --- 6. Сохранение артефакта (локально и в W&B) ---
    output_dir = Path(hydra.utils.get_original_cwd()) / "data" / cfg.data.feature_lists_path
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_filename = f"{feature_artifact_name}_top_{top_n}.txt"
    output_filepath = output_dir / output_filename
    
    with open(output_filepath, 'w') as f:
        for feature in top_features:
            f.write(f"{feature}\n")
            
    print(f"\nОтбор завершен. Список из {len(top_features)} признаков сохранен в:")
    print(output_filepath)
    
    # Логируем артефакт в W&B
    artifact_name = f"feature-list-{feature_artifact_name}-top-{top_n}"
    feature_list_artifact = wandb.Artifact(
        name=artifact_name,
        type="feature_list",
        description=f"Список топ-{top_n} признаков, отобранных из набора '{feature_artifact_name}'"
    )
    feature_list_artifact.add_file(str(output_filepath))
    run.log_artifact(feature_list_artifact)
    
    print(f"\nАртефакт со списком признаков '{artifact_name}' сохранен в W&B.")
    
    end_time = time.time()
    print(f"Общее время выполнения: {end_time - start_time:.2f} секунд.")
    run.finish()

if __name__ == "__main__":
    select_features()