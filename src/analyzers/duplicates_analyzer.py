from enum import Enum
import pandas as pd
from src.analyzers.analysis_step import AnalysisStep

class Duplicates(Enum):
    REMOVE = "REMOVE"
    AGGREGATE = "AGGREGATE"
    NO_DUPLICATES = "NO_DUPLICATES"
    
class DuplicatesSolution(Enum):
    pass

class DuplicatesAnalyzer(AnalysisStep):
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> dict:
        pass