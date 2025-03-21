# src/eda_analyzer.py
from abc import ABC, abstractmethod
import pandas as pd

class AnalysisStep(ABC):
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> dict:
        pass
    
    @abstractmethod
    def report(self) -> str:
        pass

class DataProfiler(AnalysisStep):
    def __init__(self):
        pass
    
    def analyze(self, data: pd.DataFrame) -> dict:
        pass

    def report(self) -> str:
        pass


class MissingValuesAnalyzer(AnalysisStep):
    def __init__(self, low_threshold: float = 0.05, high_threshold: float = 0.3):
        pass
    
    def analyze(self, data: pd.DataFrame) -> dict:
        pass
    
    def report(self) -> str:
        pass
    
    def _suggest_imputation_strategy(self) -> str:
        pass


class OutlierDetector(AnalysisStep):
    def __init__(self, method: str = 'IQR'):
        pass
    
    def analyze(self, data: pd.DataFrame) -> dict:
        pass
    
    def report(self) -> str:
        pass
    
    def _suggest_treatment(self) -> list:
        pass


class DistributionAnalyzer(AnalysisStep):
    def __init__(self):
        pass
    
    def analyze(self, data: pd.DataFrame) -> dict:
        pass

    def report(self) -> str:
        pass
    
    def apply_transformation(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

class CorrelationAnalyzer(AnalysisStep):
    def __init__(self):
        pass
    
    def analyze(self, data: pd.DataFrame) -> dict:
        pass

    def report(self) -> str:
        pass


class EDAAnalyzer:
    def __init__(self):
        self.steps = [
            DataProfiler(),
            MissingValuesAnalyzer(),
            OutlierDetector(),
            DistributionAnalyzer(),
            CorrelationAnalyzer()
        ]
    
    def run_analysis(self, data: pd.DataFrame) -> dict:
        pass
    
    def generate_full_report(self) -> str:
        pass
