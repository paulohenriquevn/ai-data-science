from abc import ABC, abstractmethod
import pandas as pd


class AnalysisStep(ABC):
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> dict:
        pass
