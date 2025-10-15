# src/utils.py

import os
import random
from datetime import datetime
from typing import Any

import numpy as np
import torch  # Если используется PyTorch для трансформеров

# ==================================================================================
# Воспроизводимость (Reproducibility)
# ==================================================================================

def seed_everything(seed: int) -> None:
    """
    Фиксирует сиды для всех основных источников случайности, чтобы обеспечить
    воспроизводимость экспериментов.

    Args:
        seed (int): Целое число, используемое в качестве сида.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # Сиды для PyTorch, если он используется в проекте
    # (например, для трансформерных эмбеддингов)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Некоторые операции в cuDNN могут быть недетерминированными.
            # Эти флаги пытаются это исправить, но могут замедлить обучение.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        print("PyTorch не найден. Сиды для PyTorch не установлены.")
        
    print(f"Все источники случайности зафиксированы сидом: {seed}")

# ==================================================================================
# Вспомогательные функции (Helpers)
# ==================================================================================

def get_timestamp() -> str:
    """
    Возвращает текущую временную метку в удобном для именования файлов формате.

    Returns:
        str: Строка формата 'ГГГГММДД_ЧЧММСС'.
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')


# ==================================================================================
# Функции для работы с Hydra (Опционально, но полезно)
# ==================================================================================

def get_hydra_logging_directory() -> str:
    """

    Возвращает текущую директорию логирования, созданную Hydra.
    Полезно, если нужно получить доступ к этой папке из любой части кода.
    
    Returns:
        str: Путь к директории Hydra для текущего запуска.
        
    Raises:
        ImportError: Если hydra не установлена.
        ValueError: Если функция вызвана вне Hydra-приложения.
    """
    try:
        from hydra.core.hydra_config import HydraConfig
        
        if HydraConfig.initialized():
            return HydraConfig.get().runtime.output_dir
        else:
            raise ValueError("Hydra-конфигурация не инициализирована. Запустите скрипт как Hydra-приложение.")
            
    except ImportError:
        raise ImportError("Библиотека hydra-core не установлена.")