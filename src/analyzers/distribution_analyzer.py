from enum import Enum
import pandas as pd
from src.analyzers.analysis_step import AnalysisStep

class Distributions(Enum):
    NORMAL = "NORMAL"
    LOG_NORMAL = "LOG_NORMAL"
    POISSON = "POISSON"
    BINOMIAL = "BINOMIAL"
    MULTINOMIAL = "MULTINOMIAL"
    EXPONENTIAL = "EXPONENTIAL"
    UNIFORM = "UNIFORM"
    NORMAL_COMPOSTA = "NORMAL_COMPOSTA"   

class DistributionSolution(Enum):
    pass

class DistributionAnalyzer(AnalysisStep):
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> dict:
        pass