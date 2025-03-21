from abc import ABC, abstractmethod
import pandas as pd


class FeatureEngineerStep(ABC):
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        pass
