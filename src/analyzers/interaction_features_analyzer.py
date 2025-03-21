from enum import Enum
import pandas as pd
from src.analyzers.analysis_step import AnalysisStep

class InteractionFeatures(Enum):
    POLYNOMIAL_FEATURES = "POLYNOMIAL_FEATURES"
    MULTIPLICATIVE_FEATURES = "MULTIPLICATIVE_FEATURES"
    NO_INTERACTION_FEATURES = "NO_INTERACTION_FEATURES"

class InteractionFeaturesSolution(Enum):
    pass

class InteractionFeaturesAnalyzer(AnalysisStep):
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> dict:
        pass
