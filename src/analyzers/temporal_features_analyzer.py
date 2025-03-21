from enum import Enum
import pandas as pd
from src.analyzers.analysis_step import AnalysisStep


class TemporalFeatures(Enum):
    SAZONALITY = "SAZONALITY"
    TRENDS = "TRENDS"
    NO_TEMPORAL_FEATURES = "NO_TEMPORAL_FEATURES"


class TemporalFeaturesSolution(Enum):
    pass
    
class TemporalFeaturesAnalyzer(AnalysisStep):
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> dict:
        pass
