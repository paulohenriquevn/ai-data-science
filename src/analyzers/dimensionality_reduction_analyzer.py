from enum import Enum
import pandas as pd
from src.analyzers.analysis_step import AnalysisStep

class DimensionalityReduction(Enum):
    LDA = "LDA"
    PCA = "PCA"
    UMAP = "UMAP"
    SELECTION_BY_IMPORTANCE = "SELECTION_BY_IMPORTANCE"
    NO_DIMENSIONALITY_REDUCTION = "NO_DIMENSIONALITY_REDUCTION"

class DimensionalityReductionSolution(Enum):
    pass

class DimensionalityReductionAnalyzer(AnalysisStep):

    def analyze(self, data: pd.DataFrame) -> dict:
        pass