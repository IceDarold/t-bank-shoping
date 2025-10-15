# src/tune.py

from pathlib import Path
import warnings

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import optuna
from optuna_integration import WandbCallback
import wandb

# Импортируем нашу основную функцию обучения
from src.train import train

warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)


# ==================================================================================
# Objective Function
# ==================================================================================
def objective(trial: optuna.trial.Trial, cfg: DictConfig) -> float:
    """
    "Целевая функция" для Optuna.

    На каждой итерации эта функция:
    1. Получает от Optuna `trial` с предложенными гиперпараметрами.
    2. Динамически перезаписывает этими параметрами копию основного конфига.
    3. Запускает полный пайплайн `train.py` с новым конфигом.
    4. Возвращает итоговый OOF-скор, который Optuna пытается максимизировать.
    """
    
    # Создаем копию конфига, чтобы не изменять оригинал между trials
    trial_cfg = cfg.copy()
    
    # --- Динамически предлагаем гиперпараметры из `search_space` ---
    with open_dict(trial_cfg): # Позволяем редактировать конфиг
        for param_path, search_space in cfg.tuning.search_space.items():
            param_type = search_space.type
            param_name = param_path.split('.')[-1]
            
            # Копируем параметры для suggest, удаляя наш 'type'
            suggest_params = {k: v for k, v in search_space.items() if k != 'type'}

            if param_type == 'float':
                value = trial.suggest_float(param_name, **suggest_params)
            elif param_type == 'int':
                value = trial.suggest_int(param_name, **suggest_params)
            elif param_type == 'categorical':
                value = trial.suggest_categorical(param_name, **suggest_params['choices'])
            else:
                raise ValueError(f"Неподдерживаемый тип параметра: {param_type}")
            
            # Обновляем значение в объекте конфига
            OmegaConf.update(trial_cfg, param_path, value)

    # Запускаем обучение с новыми параметрами
    try:
        oof_score = train(trial_cfg)
    except Exception as e:
        print(f"Ошибка во время trial: {e}. Optuna обработает это как неудачный запуск.")
        raise optuna.exceptions.TrialPruned()
        
    return oof_score

# ==================================================================================
# Main Tuning Function
# ==================================================================================
@hydra.main(config_path="../conf", config_name="config", version_base=None)
def tune(cfg: DictConfig) -> None:
    """
    Главный скрипт для запуска подбора гиперпараметров.
    """
    start_time = time.time()
    
    # --- 1. Инициализация ---
    print("--- Запуск подбора гиперпараметров ---")
    print("Конфигурация тюнинга:")
    print(OmegaConf.to_yaml(cfg.tuning))
    
    # Настраиваем W&B Callback для интеграции с Optuna
    wandb_callback = WandbCallback(
        metric_name=f"oof_score_{cfg.metric.main._target_.split('.')[-1].replace('Metric','')}_mean",
        wandb_kwargs={
            "project": cfg.wandb.project,
            "entity": cfg.wandb.entity,
            "job_type": "tuning"
        },
        as_multirun=True, # Логирует каждый trial как отдельный W&B run
    )

    # --- 2. Создание и запуск "исследования" Optuna ---
    study = optuna.create_study(
        direction=cfg.tuning.direction,
        study_name=f"{cfg.model._target_.split('.')[-2]}-{cfg.features.name}-tuning"
    )

    study.optimize(
        lambda trial: objective(trial, cfg),
        n_trials=cfg.tuning.n_trials,
        callbacks=[wandb_callback]
    )

    # --- 3. Анализ и вывод результатов ---
    print("\n--- Подбор гиперпараметров завершен ---")
    print(f"Количество завершенных trials: {len(study.trials)}")
    
    best_trial = study.best_trial
    print("\n--- Лучший trial ---")
    print(f"  Value (OOF Score): {best_trial.value:.5f}")
    print("  Params: ")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # --- 4. Автоматическое создание и сохранение "тюнингованного" конфига ---
    print("\n--- Создание и сохранение тюнингованного конфига модели ---")
    
    base_model_config = cfg.model.copy()
    
    # Удаляем _target_ на время, чтобы избежать проблем с OmegaConf
    target_path = base_model_config.pop('_target_')
    
    # Обновляем `params` в базовом конфиге лучшими найденными параметрами
    with open_dict(base_model_config):
        for key, value in best_trial.params.items():
            # Находим полный путь к параметру в search_space
            full_param_path = next((path for path in cfg.tuning.search_space if path.endswith(key)), None)
            if full_param_path:
                # Обновляем только `params` часть
                relative_path = ".".join(full_param_path.split('.')[1:])
                OmegaConf.update(base_model_config, relative_path, value)

    # Возвращаем _target_ на место
    base_model_config['_target_'] = target_path

    # --- 5. Сохранение артефактов ---
    # Получаем ID главного W&B run'а, который создал callback
    main_wandb_run = wandb.run or wandb.Api().run(wandb_callback.wandb_runs[0])
    
    tuned_model_name = f"{cfg.model._target_.split('.')[-2]}_tuned_{main_wandb_run.id}"
    
    # Сохраняем локально
    output_dir = Path(hydra.utils.get_original_cwd()) / "conf/model"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filepath = output_dir / f"{tuned_model_name}.yaml"
    
    with open(output_filepath, 'w') as f:
        OmegaConf.save(config=base_model_config, f=f)
    print(f"Тюнингованный конфиг сохранен локально: {output_filepath}")

    # Сохраняем как артефакт в W&B
    artifact = wandb.Artifact(
        name=f"config-{tuned_model_name}",
        type="model-config",
    )
    artifact.add_file(str(output_filepath))
    main_wandb_run.log_artifact(artifact)
    print(f"Артефакт с конфигом '{artifact.name}' сохранен в W&B.")
    print("\n💡 Теперь вы можете использовать эту модель в `train.py`, указав "
          f"`model={tuned_model_name}`")
    
    main_wandb_run.finish()

if __name__ == "__main__":
    tune()