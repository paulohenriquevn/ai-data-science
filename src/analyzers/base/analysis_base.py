from abc import ABC, abstractmethod
import pandas as pd


class AnalysisStep(ABC):
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> dict:
        pass

class PlanStep(ABC):
    @abstractmethod
    def generate(self, data: pd.DataFrame) -> dict:
        pass

class ExecutionStep(ABC):
    @abstractmethod
    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        pass
    
