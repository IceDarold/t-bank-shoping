# README: Архитектура Модуля Метрик (`src/metrics/`)

Этот документ описывает внутреннее устройство модуля метрик. Его цель — объяснить разработчикам, как существующие метрики интегрированы в фреймворк и как можно быстро добавить любую новую метрику.

## 1. 🎯 Философия Архитектуры: "Контракт" и Гибкость

Как и в модуле моделей, мы используем принцип **абстракции**, чтобы пайплайн обучения (`train.py`) был независим от конкретной реализации метрики.

1.  **Интерфейс Метрики (`src/metrics/base.py`):** Это "контракт", который определяет, что любая метрика в нашей системе должна быть *вызываемым* объектом (callable), принимающим `y_true` и `y_pred` и возвращающим число.
2.  **Классы-Обертки (`src/metrics/roc_auc.py`, `src/metrics/rmse.py` и т.д.):** Каждый такой класс "оборачивает" одну функцию метрики (чаще всего из `scikit-learn`) и реализует "контракт".

Это позволяет нам легко переключаться между `ROC AUC` и `F1-Score`, или `RMSE` и `MAPE`, просто меняя одну строку в конфигурации, без каких-либо изменений в коде `train.py`.

## 2. 📁 Структура Папок: Одна Метрика — Один Файл

Каждая метрика вынесена в свой собственный файл, что делает систему чистой, модульной и легко расширяемой.

```
src/
└── metrics/
    ├── __init__.py
    ├── base.py       # Абстрактный базовый класс MetricInterface
    ├── roc_auc.py    # Реализация для ROC AUC
    ├── f1_score.py   # Реализация для F1-Score
    ├── rmse.py       # Реализация для RMSE
    └── ...           # и другие файлы с метриками
```

### `MetricInterface` (`src/metrics/base.py`)

"Контракт" предельно прост:

```python
class MetricInterface(ABC):
    @abstractmethod
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Вычисляет значение метрики."""
        pass
```
Реализация метода `__call__` делает экземпляры наших классов метрик похожими на функции, что позволяет писать в `train.py` очень чистый код: `score = metric(y_true, y_pred)`.

---

## 3. 🚀 Туториал: Добавление Новой (Кастомной) Метрики

Предположим, для соревнования вам нужна специфическая метрика, которой нет в `scikit-learn`, например, **F0.5-score** (F-beta score, где `beta=0.5`), которая придает больше веса `precision`, чем `recall`.

**Шаг 1: Создайте файл `src/metrics/fbeta.py`**

```python
# src/metrics/fbeta.py
import numpy as np
from sklearn.metrics import fbeta_score
from .base import MetricInterface

class FBetaMetric(MetricInterface):
    """
    Вычисляет F-beta меру.
    
    Это обобщенная F1-мера, где `beta` контролирует баланс
    между precision и recall.
    - beta < 1: больше веса у precision.
    - beta > 1: больше веса у recall.
    """
    def __init__(self, beta: float, threshold: float = 0.5, average: str = 'binary'):
        if beta <= 0:
            raise ValueError("Параметр 'beta' должен быть положительным.")
        self.beta = beta
        self.threshold = threshold
        self.average = average

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred_class = (y_pred > self.threshold).astype(int)
        return fbeta_score(
            y_true, 
            y_pred_class, 
            beta=self.beta, 
            average=self.average,
            zero_division=0
        )
```

**Шаг 2: Создайте YAML-конфиг в `conf/metric/`**

Создайте файл `conf/metric/f05_score.yaml`.

```yaml
# conf/metric/f05_score.yaml

# Указываем Hydra путь к нашему новому классу
_target_: src.metrics.fbeta.FBetaMetric

# --- Передаем параметры в конструктор __init__ ---
beta: 0.5
threshold: 0.5
average: 'binary'
```

**Шаг 3: Готово! Используйте новую метрику**

Теперь вы можете использовать вашу новую метрику где угодно.

*   **Как основную метрику:**
    ```bash
    python src/train.py metric=f05_score
    ```

*   **Как дополнительную метрику для логирования:**
    В вашем "сборном" конфиге, например `conf/metric/classification.yaml`, добавьте ее в секцию `additional`:
    ```yaml
    additional:
      # ...
      - _target_: src.metrics.fbeta.FBetaMetric
        name: "f0.5_score"
        beta: 0.5
    ```

Эта архитектура позволяет вам быстро и безболезненно расширять библиотеку метрик, адаптируя ее под требования любого соревнования, даже с самыми экзотическими функциями оценки.