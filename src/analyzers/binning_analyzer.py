from enum import Enum
import pandas as pd
from src.analyzers.analysis_step import AnalysisStep

class Binning(Enum):
    QUANTILE_BINNING = "QUANTILE_BINNING"
    EQUAL_WIDTH_BINNING = "EQUAL_WIDTH_BINNING"
    NO_BINNING = "NO_BINNING"
    
class BinningSolution(Enum):
    pass

class BinningAnalyzer(AnalysisStep):
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> dict:
        pass