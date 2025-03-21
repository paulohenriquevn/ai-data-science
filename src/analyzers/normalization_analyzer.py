from enum import Enum
import pandas as pd
from src.analyzers.analysis_step import AnalysisStep

class Normalization(Enum):  
    MIN_MAX_SCALER = "MIN_MAX_SCALER"
    STANDARD_SCALER = "STANDARD_SCALER"
    ROBUST_SCALER = "ROBUST_SCALER"  
    
class NormalizationSolution(Enum):
    pass

class NormalizationAnalyzer(AnalysisStep):
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> dict:
        pass