import pandas as pd
from abc import ABC, abstractmethod


class GroupStep:
    @abstractmethod
    def run(self, data: pd.DataFrame) -> dict:
        pass


class AnalysisStep(ABC):
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> dict:
        pass
    
    
class ChoiceStep(ABC):
    @abstractmethod
    def answer(self, data: pd.DataFrame) -> dict:
        pass

class PlanStep(ABC):
    @abstractmethod
    def generate(self, analysis_output: list) -> list:
        pass

class ExecutionStep(ABC):
    @abstractmethod
    def execute(self, df: pd.DataFrame, plan: list) -> pd.DataFrame:
        pass


class ReportStep(ABC):
    @abstractmethod
    def generate(self, plan: list) -> list:
        pass