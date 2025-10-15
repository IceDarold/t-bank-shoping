# src/features/__init__.py

import os
import importlib.util
import sys
from pathlib import Path

# --- Полностью автоматический механизм регистрации всех генераторов ---

def register_features():
    """
    Рекурсивно обходит все .py файлы в пакете 'src.features',
    динамически импортирует их, чтобы все классы FeatureGenerator
    стали известны Python и, следовательно, Hydra.
    """
    
    # 1. Получаем путь к текущей папке (src/features)
    package_path = Path(__file__).parent.resolve()
    
    # 2. Получаем имя пакета (например, 'src.features')
    # Это нужно для правильного формирования полных имен модулей.
    src_path = package_path.parent.resolve()
    package_name = f"{src_path.name}.{package_path.name}"

    print(f"--- Регистрация генераторов признаков из пакета '{package_name}' ---")

    # 3. Рекурсивно обходим все директории и файлы
    for root, dirs, files in os.walk(package_path):
        for filename in files:
            if filename.endswith('.py') and filename != '__init__.py':
                
                # 4. Формируем полный путь к файлу и его модульное имя
                file_path = Path(root) / filename
                # Получаем относительный путь от папки 'features'
                relative_path = file_path.relative_to(package_path)
                
                # Превращаем путь в модульное имя (e.g., 'numerical.scaling')
                module_subpath = ".".join(relative_path.with_suffix('').parts)
                module_name = f"{package_name}.{module_subpath}"

                try:
                    # 5. Динамически импортируем модуль
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                    print(f"  [OK] Модуль '{module_name}' успешно зарегистрирован.")
                except Exception as e:
                    print(f"  [ERROR] Ошибка при регистрации модуля '{module_name}': {e}")

# 6. Запускаем регистрацию один раз при импорте пакета 'features'
register_features()

# 7. Для удобства можно оставить импорт базового класса
from .base import FeatureGenerator