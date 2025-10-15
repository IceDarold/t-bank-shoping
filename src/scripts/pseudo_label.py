# src/pseudo_label.py

import os
from pathlib import Path
import shutil
import time

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf, open_dict
import wandb

# Импортируем `main` функции из наших скриптов, чтобы вызывать их напрямую
from src.make_features import make_features
from src.train import train
from src import utils


def get_latest_hydra_run_path() -> Path:
    """Находит путь к самой последней созданной директории Hydra в `outputs`."""
    outputs_path = Path.cwd() / "outputs"
    # Ищем по всем возможным путям (multirun, simple run)
    all_runs = sorted(outputs_path.glob("**/.hydra"), key=os.path.getmtime)
    if not all_runs:
        raise FileNotFoundError("Не найдено ни одного запуска Hydra в папке 'outputs'.")
    # Возвращаем родительскую директорию от `.hydra`
    return all_runs[-1].parent


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def pseudo_label(cfg: DictConfig) -> None:
    """
    Главный пайплайн для выполнения псевдо-лейблинга.
    """
    start_time = time.time()
    pl_cfg = cfg.pseudo_labeling
    
    # === 1. Инициализация главного W&B run'а ===
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=f"pseudo-labeling-{pl_cfg.name}-{utils.get_timestamp()}",
        job_type="pseudo-labeling",
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    
    print("--- Запуск пайплайна Псевдо-лейблинга ---")
    print(OmegaConf.to_yaml(pl_cfg))
    
    # --- Подготовка ---
    # Создаем временную папку для артефактов этого процесса
    temp_dir = Path(pl_cfg.temp_data_path)
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    data_path = Path(hydra.utils.get_original_cwd()) / "data"
    original_train_df = pd.read_csv(data_path / cfg.data.processed_path / cfg.data.train_file)
    original_test_df = pd.read_csv(data_path / cfg.data.processed_path / cfg.data.test_file)

    # --- РАУНД 0: Обучение на исходных данных ---
    print("\n\n--- РАУНД 0: Обучение на исходных данных ---")
    
    # Формируем конфиг для первого запуска train
    train_cfg_round0 = cfg.copy()
    with open_dict(train_cfg_round0): # Делаем конфиг редактируемым
        train_cfg_round0.experiment = cfg.experiments[pl_cfg.training_experiment_config]

    # Запускаем обучение
    train(train_cfg_round0)
    
    # Находим артефакты этого запуска
    round0_output_path = get_latest_hydra_run_path()
    current_test_preds_path = round0_output_path / "submission.csv"
    current_train_df = original_train_df.copy()

    # --- ОСНОВНОЙ ЦИКЛ ПСЕВДО-ЛЕЙБЛИНГА ---
    for i in range(pl_cfg.num_rounds):
        round_num = i + 1
        print(f"\n\n--- РАУНД {round_num}/{pl_cfg.num_rounds}: Начало итерации ---")

        # --- Шаг 1: Создание псевдо-датасета ---
        print(f"--- Шаг {round_num}.1: Создание псевдо-датасета ---")
        test_preds_df = pd.read_csv(current_test_preds_path)
        
        thresholds = pl_cfg.confidence_thresholds[i]
        confident_preds = test_preds_df[
            (test_preds_df[cfg.globals.target_col] > thresholds.high) | 
            (test_preds_df[cfg.globals.target_col] < thresholds.low)
        ].copy()
        
        confident_preds[cfg.globals.target_col] = (confident_preds[cfg.globals.target_col] > 0.5).astype(int)
        
        pseudo_labeled_data = pd.merge(original_test_df, confident_preds, on=cfg.globals.id_col)
        new_train_df = pd.concat([original_train_df, pseudo_labeled_data], ignore_index=True)
        
        new_train_path = temp_dir / f"train_pseudo_round_{round_num}.csv"
        new_train_df.to_csv(new_train_path, index=False)
        print(f"Создан новый трейн-сет: {new_train_path.name} (shape: {new_train_df.shape})")
        print(f"Добавлено {len(pseudo_labeled_data)} псевдо-меток.")
        run.log({"pseudo_labels_added": len(pseudo_labeled_data), "round": round_num})
        
        # --- Шаг 2: Перегенерация признаков ---
        print(f"--- Шаг {round_num}.2: Перегенерация признаков ---")
        fe_cfg = cfg.copy()
        new_fe_name = f"{pl_cfg.feature_engineering_config}_pl_round_{round_num}"
        with open_dict(fe_cfg):
            fe_cfg.feature_engineering = cfg.feature_engineering[pl_cfg.feature_engineering_config]
            fe_cfg.feature_engineering.name = new_fe_name
            # Подменяем пути к данным
            fe_cfg.data.train_file = str(new_train_path.name)
            fe_cfg.data.processed_path = str(temp_dir.relative_to(data_path))

        make_features(fe_cfg)

        # --- Шаг 3: Переобучение модели ---
        print(f"--- Шаг {round_num}.3: Переобучение модели ---")
        train_cfg_roundN = cfg.copy()
        with open_dict(train_cfg_roundN):
            train_cfg_roundN.experiment = cfg.experiments[pl_cfg.training_experiment_config]
            # Подменяем имя набора признаков на новый, сгенерированный на шаге 2
            train_cfg_roundN.feature_engineering.name = new_fe_name

        oof_score = train(train_cfg_roundN)
        run.log({"oof_score": oof_score, "round": round_num})
        
        # --- Шаг 4: Обновление путей для следующей итерации ---
        roundN_output_path = get_latest_hydra_run_path()
        current_test_preds_path = roundN_output_path / "submission.csv"

    # --- Финализация ---
    print("\n\n--- Псевдо-лейблинг завершен! ---")
    final_submission_path = current_test_preds_path
    
    # Копируем финальный сабмишен в корень проекта для удобства
    final_dest_path = Path.cwd() / f"submission_pseudo_{pl_cfg.name}.csv"
    shutil.copy(final_submission_path, final_dest_path)
    
    print(f"Финальный сабмишен скопирован в: {final_dest_path}")
    
    # Логируем финальный сабмишен как артефакт главного run'а
    final_artifact = wandb.Artifact(name=f"submission_{pl_cfg.name}", type="submission")
    final_artifact.add_file(str(final_dest_path))
    run.log_artifact(final_artifact)
    
    end_time = time.time()
    print(f"Общее время выполнения: {end_time - start_time:.2f} секунд.")
    run.finish()

if __name__ == "__main__":
    pseudo_label()