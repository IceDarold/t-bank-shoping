# Руководство по Созданию Генераторов Признаков

Этот документ предназначен для разработчиков, которые хотят расширить возможности фреймворка, добавляя свои собственные, кастомные генераторы признаков.

## 1. 🎯 Архитектура Генераторов

В основе нашей системы лежит простой, но мощный дизайн:

1.  **Абстрактный Базовый Класс:** Все генераторы должны наследовать от `src.features.base.FeatureGenerator`. Этот класс- "контракт" требует реализации методов `fit` и `transform`.

2.  **Стратегия Обучения (`fit_strategy`):** Каждый генератор имеет атрибут `fit_strategy`, который сообщает оркестратору `make_features.py`, на каких данных его обучать.
    *   `fit_strategy = "train_only"` (по умолчанию): Метод `.fit()` будет вызван **только на обучающей выборке**. Используется для большинства генераторов, особенно для supervised.
    *   `fit_strategy = "combined"`: Метод `.fit()` будет вызван на **объединенных данных (train + test)**. Используется для unsupervised-генераторов, которым нужно выучить глобальную структуру данных (например, `AggregationGenerator`, `AutoencoderFeatureGenerator`).

3.  **Автоматическая Регистрация:** Вам **не нужно нигде вручную импортировать** ваш новый генератор. Система автоматически сканирует все `.py` файлы в `src/features/` (включая все подпапки) и делает их доступными для Hydra.

---

## 2. 🚀 Туториал: Создание Вашего Первого Генератора

Давайте создадим простой генератор, который вычисляет отношение между двумя колонками.

**Шаг 1: Создайте файл**

Лучшее место для ваших личных, специфичных для проекта генераторов — это папка `src/features/custom/`.

Создайте файл: `src/features/custom/ratio_generator.py`

**Шаг 2: Напишите код**

```python
# src/features/custom/ratio_generator.py

import pandas as pd
from ..base import FeatureGenerator # 👈 Наследуемся от базового класса

class RatioFeatureGenerator(FeatureGenerator):
    """
    Создает новый признак как отношение двух колонок (col_a / col_b).
    """
    # Этот генератор - unsupervised и stateless. Стратегия по умолчанию
    # "train_only" подходит, так как .fit() все равно ничего не делает.
    
    def __init__(self, name: str, col_a: str, col_b: str, epsilon: float = 1e-6):
        super().__init__(name)
        self.col_a = col_a
        self.col_b = col_b
        self.epsilon = epsilon

    def fit(self, data: pd.DataFrame) -> None:
        # Этот генератор не требует обучения, поэтому fit пустой.
        pass

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        new_col_name = f"{self.name}_{self.col_a}_div_{self.col_b}"
        df[new_col_name] = df[self.col_a] / (df[self.col_b] + self.epsilon)
        return df
```

**Шаг 3: Используйте его в вашем конвейере**

Теперь откройте любой `.yaml` файл в `conf/feature_engineering/` и добавьте новый шаг в `pipeline`, указав **полный путь** к вашему новому классу.

```yaml
# conf/feature_engineering/my_custom_pipeline.yaml

name: "my_custom_pipeline"

pipeline:
  # ... другие шаги ...

  # 👇 Наш новый кастомный шаг
  - _target_: src.features.custom.ratio_generator.RatioFeatureGenerator
    name: "income_to_age_ratio"
    col_a: "income"
    col_b: "age"
```

**Шаг 4: Запустите `make_features`**

```bash
python src/make_features.py feature_engineering=my_custom_pipeline
```

**Готово!** Скрипт `make_features.py` автоматически обнаружит, инстанциирует и выполнит ваш новый генератор.

---

## 3. 📚 Справочник по "Ядерным" Генераторам

Чтобы увидеть примеры более сложных генераторов и понять, как они работают, обратитесь к документации по стандартным компонентам:

> 📖 **[Полный Справочник по Генераторам Признаков (docs/README.md)](../../docs/feature_generators/README.md)**