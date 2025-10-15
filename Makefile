# ==============================================================================
# Makefile для Фреймворка ML-экспериментов
# ==============================================================================
# Этот Makefile служит для упрощения запуска основных команд пайплайна.
#
# Основные переменные:
#   E = Имя `experiment` конфига (из `conf/experiment/`)
#   I = Имя `inference` конфига (из `conf/inference/`)
#   S = Имя `stacking` конфига (из `conf/stacking/`)
#   T = Имя `tuning` конфига (из `conf/tuning/`)
#   SEL = Имя `selection` конфига (из `conf/selection/`)
#
# Вы можете переопределять их при запуске, например:
# `make train E=exp002_catboost`

# --- Переменные с значениями по умолчанию ---
E ?= exp001_titanic_lgbm       # Базовый эксперимент для примера
I ?= inf_exp001               # Базовый инференс для примера
S ?= titanic_stack             # Базовый стекинг для примера
T ?= titanic_lgbm              # Базовый тюнинг для примера
SEL ?= default                # Базовый конфиг отбора признаков

.PHONY: help install features select train fulltrain tune stack predict pseudo clean

# ==============================================================================
# --- Основная команда: Справка ---
# ==============================================================================

help:
	@echo "=============================================================================="
	@echo "  Фреймворк для ML-экспериментов: Справочник по командам"
	@echo "=============================================================================="
	@echo ""
	@echo "  НАСТРОЙКА:"
	@echo "    make install          - Установить все зависимости проекта через Poetry."
	@echo ""
	@echo "  ОСНОВНОЙ WORKFLOW:"
	@echo "    make features E=<exp> - Сгенерировать признаки для эксперимента (по умолч.: $(E))."
	@echo "    make select E=<exp>   - Выполнить отбор признаков на основе эксперимента (по умолч.: $(E))."
	@echo "    make train E=<exp>    - Обучить модель на CV для эксперимента (по умолч.: $(E))."
	@echo "    make fulltrain E=<exp>- Обучить модель на 100% данных для эксперимента (по умолч.: $(E))."
	@echo ""
	@echo "  ПРОДВИНУТЫЕ ТЕХНИКИ:"
	@echo "    make tune T=<tune> E=<exp> - Запустить подбор гиперпараметров (по умолч. T=$(T), E=$(E))."
	@echo "    make stack S=<stack>  - Запустить стекинг (по умолч.: $(S))."
	@echo "    make predict I=<inf>  - Сделать инференс по результатам эксперимента (по умолч.: $(I))."
	@echo "    make pseudo           - Запустить пайплайн псевдо-лейблинга (использует конфиг по умолчанию)."
	@echo ""
	@echo "  ОБСЛУЖИВАНИЕ:"
	@echo "    make clean            - Удалить временные файлы Python (__pycache__, *.pyc)."
	@echo ""
	@echo "  ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ:"
	@echo "    make train E=exp002_catboost"
	@echo "    make select E=exp002_catboost selection.top_n=300"
	@echo "    make tune T=catboost_search E=exp002_catboost"
	@echo "=============================================================================="


# ==============================================================================
# --- Команды ---
# ==============================================================================

# --- Настройка ---
install:
	@echo ">>> Установка зависимостей..."
	@if command -v poetry >/dev/null 2>&1; then \
		echo ">>> Используется Poetry..."; \
		poetry install; \
	else \
		echo ">>> Poetry не найден. Используется pip..."; \
		pip install -r requirements.txt; \
	fi
	@echo ">>> ГОТОВО. Не забудьте выполнить 'wandb login', если делаете это впервые."


# --- Основной Workflow ---

# 1. Генерация признаков
features:
	@echo ">>> Генерация признаков для эксперимента: $(E)..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run python -m src.scripts.make_features experiment=$(E); \
	else \
		python -m src.scripts.make_features experiment=$(E); \
	fi

# 2. Отбор признаков
select:
	@echo ">>> Отбор признаков на основе эксперимента: $(E) с конфигом отбора $(SEL)..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run python -m src.scripts.select_features experiment=$(E) selection=$(SEL); \
	else \
		python -m src.scripts.select_features experiment=$(E) selection=$(SEL); \
	fi

# 3. Обучение на CV
train:
	@echo ">>> Обучение (CV) для эксперимента: $(E)..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run python -m src.scripts.train experiment=$(E); \
	else \
		python -m src.scripts.train experiment=$(E); \
	fi

# 3b. Обучение на 100% данных
fulltrain:
	@echo ">>> Обучение (на 100% данных) для эксперимента: $(E)..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run python -m src.scripts.train experiment=$(E) training.full_data=true; \
	else \
		python -m src.scripts.train experiment=$(E) training.full_data=true; \
	fi


# --- Продвинутые Техники ---

# 4. Подбор гиперпараметров
tune:
	@echo ">>> Подбор гиперпараметров с конфигом '$(T)' для эксперимента '$(E)'..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run python -m src.scripts.tune experiment=$(E) tuning=$(T); \
	else \
		python -m src.scripts.tune experiment=$(E) tuning=$(T); \
	fi

# 5. Стекинг
stack:
	@echo ">>> Запуск стекинга с конфигурацией: $(S)..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run python -m src.scripts.stack stacking=$(S); \
	else \
		python -m src.scripts.stack stacking=$(S); \
	fi

# 6. Инференс
predict:
	@echo ">>> Инференс с конфигурацией: $(I)..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run python -m src.scripts.predict inference=$(I); \
	else \
		python -m src.scripts.predict inference=$(I); \
	fi

# 7. Псевдо-лейблинг
pseudo:
	@echo ">>> Запуск пайплайна псевдо-лейблинга..."
	@if command -v poetry >/dev/null 2>&1; then \
		poetry run python -m src.scripts.pseudo_label; \
	else \
		python -m src.scripts.pseudo_label; \
	fi


# --- Обслуживание ---
clean:
	@echo ">>> Очистка временных файлов Python..."
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	@echo ">>> Очистка завершена."