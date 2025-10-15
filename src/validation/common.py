from .base import BaseSplitter
from sklearn.model_selection._split import BaseCrossValidator

class SklearnSplitterWrapper(BaseSplitter):
    def __init__(self, splitter_class: str, **kwargs):
        # Динамически импортируем класс из sklearn
        module_path, class_name = splitter_class.rsplit('.', 1)
        module = __import__(module_path, fromlist=[class_name])
        splitter_constructor = getattr(module, class_name)
        
        self.splitter: BaseCrossValidator = splitter_constructor(**kwargs)

    def split(self, data, y, groups=None):
        return self.splitter.split(data, y, groups)

    def get_n_splits(self, data=None, y=None, groups=None):
        return self.splitter.get_n_splits(data, y, groups)