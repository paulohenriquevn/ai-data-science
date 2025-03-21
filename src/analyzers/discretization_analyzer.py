from enum import Enum
import pandas as pd
from src.analyzers.analysis_step import AnalysisStep

class Discretization(Enum):
    DISCRETIZATION_EQUAL_WIDTH = "DISCRETIZATION_EQUAL_WIDTH"
    DISCRETIZATION_QUANTILE = "DISCRETIZATION_QUANTILE"
    NO_DISCRETIZATION = "NO_DISCRETIZATION"

class DiscretizationSolution(Enum):
    pass

class DiscretizationAnalyzer(AnalysisStep):

    def analyze(self, data: pd.DataFrame) -> dict:
        pass