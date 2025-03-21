from enum import Enum
import pandas as pd
from src.analyzers.analysis_step import AnalysisStep

class Outliers(Enum):
    IQR_10 = "IQR_10"
    IQR_20 = "IQR_20"
    IQR_30 = "IQR_30"
    Z_SCORE_3 = "Z_SCORE_3"
    Z_SCORE_5 = "Z_SCORE_5"
    Z_SCORE_10 = "Z_SCORE_10"
    IMPACTO_MODERADO = "IMPACTO_MODERADO"
    IMPACTO_FORTE = "IMPACTO_FORTE"
    IMPACTO_EXTREMO = "IMPACTO_EXTREMO"
    NO_OUTLIERS = "NO_OUTLIERS"


class OutliersSolution(Enum):
    pass

class OutliersAnalyzer(AnalysisStep):
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> dict:
        pass