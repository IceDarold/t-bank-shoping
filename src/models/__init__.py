# src/models/__init__.py

from .lgbm import LGBMModel
from .xgb import XGBModel
from .catboost import CatBoostModel

# Это позволит в будущем делать так: from src.models import LGBMModel
# Вместо: from src.models.lgbm import LGBMModel