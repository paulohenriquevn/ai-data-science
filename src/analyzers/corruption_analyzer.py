from enum import Enum
import pandas as pd
from src.analyzers.analysis_step import AnalysisStep


class Corruption(Enum):
    CORRECT = "CORRECT"
    NO_CORRECT = "NO_CORRECT"
    NO_CORRUPTION = "NO_CORRUPTION"
    
class CorruptionSolution(Enum):
    pass

class CorruptionAnalyzer(AnalysisStep):
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> dict:
        pass