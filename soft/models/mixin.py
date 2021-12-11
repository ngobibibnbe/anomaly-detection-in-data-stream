from abc import abstractmethod
from pysad.core.base_model import BaseModel


class ModelMixin:
    @abstractmethod
    def evaluate(self, X, y, metric):
        pass
