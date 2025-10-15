# README: Руководство по Конвейерам Генерации Признаков (`conf/feature_engineering/`)

Эта директория — ваш "цех по производству признаков". Каждый `.yaml` файл здесь представляет собой **конвейер (pipeline)** — пошаговый "рецепт" для создания полноценного набора признаков из сырых данных.

Скрипт `make_features.py` берет один из этих файлов и последовательно выполняет все описанные в нем шаги.

## 1. 🎯 Философия: От простого к сложному

Идея конвейера — выполнять шаги в логической последовательности. Вы начинаете с простых преобразований, а затем на их основе строите более сложные.

**Пример логики конвейера:**
1.  Сначала создаем **базовые признаки** из дат (`DatetimeFeatureGenerator`).
2.  Затем создаем **агрегации**, используя новые признаки (например, группируем по месяцам).
3.  Затем **кодируем** все категориальные признаки (`TargetEncoder`).
4.  В конце, на основе всех уже созданных числовых признаков, обучаем **автоэнкодер** для получения мета-признаков.

## 2. ⚙️ Анатомия Файла Конвейера

Каждый `.yaml` файл в этой папке имеет две основные секции: `name` и `pipeline`.

**Пример: `conf/feature_engineering/v1_baseline.yaml`**
```yaml
# 1. Имя набора признаков (и артефакта W&B)
# Это имя будет использоваться в именах файлов .parquet и в W&B.
name: "v1_baseline"

# 2. Список шагов конвейера
# `make_features.py` будет выполнять эти шаги последовательно, сверху вниз.
pipeline:
  # --- ШАГ 1: Извлечь компоненты из даты ---
  - _target_: src.features.datetime.DatetimeFeatureGenerator
    name: "date_features"
    cols: ["transaction_date"]
    components: ["weekday", "month", "year"]

  # --- ШАГ 2: Создать флаги для пропусков ---
  - _target_: src.features.numerical.flags.IsNullIndicator
    name: "null_flags"
    cols: ["user_age", "user_city"]
    
  # --- ШАГ 3: Закодировать категории ---
  - _target_: src.features.categorical.nominal.CountFrequencyEncoderGenerator
    name: "count_encode_cats"
    cols: ["user_city", "product_category"]
    normalize: false
```

### Ключевые Элементы Шага

*   `_target_`: **Обязательный.** Полный путь к Python-классу `FeatureGenerator`, который нужно использовать на этом шаге.
*   `name`: **Обязательный.** Уникальное имя для этого шага, которое будет отображаться в логах.
*   **Остальные параметры** (`cols`, `components`, `normalize` и т.д.): Это параметры, которые передаются в `__init__` конструктор соответствующего класса-генератора.

---

## 3. 🚀 Практическое Руководство: Создание нового набора признаков

Предположим, вы хотите создать новую версию признаков (`v2`), добавив к базовому набору (`v1`) агрегации.

**Шаг 1: Скопируйте существующий конфиг**
Скопируйте `v1_baseline.yaml` в новый файл `v2_with_aggs.yaml`.

**Шаг 2: Измените имя**
В новом файле измените `name`, чтобы избежать перезаписи старых артефактов.
```yaml
name: "v2_with_aggs"
```

**Шаг 3: Добавьте новый шаг в `pipeline`**
Добавьте в конец списка `pipeline` новый блок для `AggregationGenerator`.

**`conf/feature_engineering/v2_with_aggs.yaml`:**
```yaml
name: "v2_with_aggs"

pipeline:
  # ... (все шаги из v1_baseline.yaml остаются здесь) ...
  - _target_: src.features.datetime.DatetimeFeatureGenerator
    # ...
  - _target_: src.features.numerical.flags.IsNullIndicator
    # ...
  - _target_: src.features.categorical.nominal.CountFrequencyEncoderGenerator
    # ...

  # --- НОВЫЙ ШАГ 4: Добавляем агрегации ---
  - _target_: src.features.aggregation.AggregationGenerator
    name: "user_aggregations"
    group_keys: ["user_id"]
    group_values: ["transaction_amount"]
    agg_funcs: ["mean", "sum", "std"]
```

**Шаг 4: Запустите генерацию**
Теперь запустите `make_features.py`, указав ему новый "рецепт":
```bash
python src/make_features.py feature_engineering=v2_with_aggs
```
Или через `Makefile`, если вы его настроили для передачи `feature_engineering`.

**Результат:**
*   Будет выполнен полный конвейер из 4 шагов.
*   В `data/03_features/` появятся файлы `train_v2_with_aggs.parquet` и `test_v2_with_aggs.parquet`.
*   В W&B будет создан новый артефакт `v2_with_aggs`.

## 4. 📚 Справочник по Генераторам

Этот фреймворк поставляется с огромной библиотекой готовых генераторов. Чтобы понять, какой генератор за что отвечает и какие у него есть параметры, обратитесь к детальной документации:

> 📖 **[Руководство для Разработчика Признаков (src/features/README.md)](../src/features/README.md)**

Эта система конвейеров позволяет вам итеративно и воспроизводимо создавать сложные наборы признаков, просто добавляя и настраивая "кирпичики" в YAML, без необходимости трогать основной код.