from enum import Enum
import pandas as pd
from src.analyzers.analysis_step import AnalysisStep

class Correlation(Enum):    
    CORRELATION_HIGH = "CORRELATION_HIGH"
    CORRELATION_MEDIUM = "CORRELATION_MEDIUM"
    CORRELATION_LOW = "CORRELATION_LOW"
    NO_CORRELATION = "NO_CORRELATION"

class CorrelationSolution(Enum):
    pass


class CorrelationAnalyzer(AnalysisStep):
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> dict:
        pass
    