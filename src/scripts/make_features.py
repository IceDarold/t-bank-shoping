# src/make_features.py

from pathlib import Path
import time

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
import wandb

# ==================================================================================
# ❗️ Важно: Этот импорт запускает автоматическую регистрацию всех генераторов.
import src.features
# ==================================================================================

from src.features.base import FeatureGenerator
from src import utils


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def make_features(cfg: DictConfig) -> None:
    """
    Главный скрипт-оркестратор для конвейера генерации признаков.

    1. Загружает данные.
    2. Последовательно применяет генераторы признаков.
    3. Сохраняет итоговые наборы признаков локально.
    4. Логирует (загружает) сохраненные признаки как версионированный
       артефакт в Weights & Biases.
    """
    start_time = time.time()
    
    print("--- Запуск конвейера генерации признаков ---")
    feature_set_name = cfg.feature_engineering.name
    print(f"Имя набора признаков (и артефакта): {feature_set_name}")
    print("---------------------------------------------")
    
    # --- 1. Загрузка исходных данных ---
    data_path = Path(hydra.utils.get_original_cwd()) / "data"
    
    processed_path = data_path / cfg.data.processed_path
    train_df = pd.read_csv(processed_path / cfg.data.train_file)
    test_df = pd.read_csv(processed_path / cfg.data.test_file)
    
    combined_df = pd.concat([train_df, test_df], ignore_index=True, sort=False)
    
    print(f"Загружены данные: train.shape={train_df.shape}, test.shape={test_df.shape}")

    # --- 2. Последовательное выполнение шагов конвейера ---
    pipeline_config = cfg.feature_engineering.pipeline
                             
    for step_cfg in tqdm(pipeline_config, desc="Выполнение шагов конвейера"):
        step_start_time = time.time()
        
        generator: FeatureGenerator = hydra.utils.instantiate(step_cfg)
        print(f"\n>> Шаг: {generator.name} (Класс: {generator.__class__.__name__})")
        
        if generator.fit_strategy == "combined":
            print("   Стратегия: 'combined'. Обучение на объединенных данных...")
            fit_data = combined_df
        else: # "train_only"
            print("   Стратегия: 'train_only'. Обучение на трейне...")
            fit_data = train_df
            
        generator.fit(fit_data)
            
        print("   Применение трансформации к train...")
        train_df = generator.transform(train_df)
        print("   Применение трансформации к test...")
        test_df = generator.transform(test_df)
        
        step_end_time = time.time()
        print(f"   Шаг выполнен за {step_end_time - step_start_time:.2f} секунд.")
        print(f"   Новые размеры: train.shape={train_df.shape}, test.shape={test_df.shape}")

    # --- 3. Сохранение локальных файлов ---
    features_path = data_path / cfg.data.features_path
    features_path.mkdir(parents=True, exist_ok=True)
    
    train_output_path = features_path / f"train_{feature_set_name}.parquet"
    test_output_path = features_path / f"test_{feature_set_name}.parquet"
    
    print("\n--- Сохранение локальных файлов признаков ---")
    train_df.to_parquet(train_output_path)
    test_df.to_parquet(test_output_path)
    print(f"Train сохранен в: {train_output_path}")
    print(f"Test сохранен в:  {test_output_path}")

    # --- 4. Логирование артефактов в W&B ---
    print("\n--- Сохранение артефактов в W&B ---")
    
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=f"make_features-{feature_set_name}-{utils.get_timestamp()}",
        job_type="feature-engineering",
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    
    artifact = wandb.Artifact(
        name=feature_set_name,
        type='dataset',
        description=f"Набор признаков, сгенерированный по конфигу '{feature_set_name}'",
        metadata={"feature_engineering_config": OmegaConf.to_container(cfg.feature_engineering)}
    )
    
    # Добавляем сохраненные файлы в артефакт
    artifact.add_file(str(train_output_path), name=f"train_{feature_set_name}.parquet")
    artifact.add_file(str(test_output_path), name=f"test_{feature_set_name}.parquet")
    
    run.log_artifact(artifact)
    run.finish()

    end_time = time.time()
    print("---------------------------------------------")
    print(f"Конвейер и логирование артефакта успешно завершены за {end_time - start_time:.2f} секунд.")
    print(f"Артефакт '{feature_set_name}' доступен в проекте W&B.")
    print("---------------------------------------------")


if __name__ == "__main__":
    make_features()