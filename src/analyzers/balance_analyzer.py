from enum import Enum
import pandas as pd
from src.analyzers.analysis_step import AnalysisStep

class Balance(Enum):
    BALANCE_MAIOR_80_UNDERSAMPLING = "BALANCE_MAIOR_80_UNDERSAMPLING"
    BALANCE_MAIOR_80_ENSEMBLE = "BALANCE_ENTRE_80_ENSEMBLE"
    BALANCE_MENOR_30_SMOTE = "BALANCE_MENOR_30_SMOTE"
    BALANCE_MENOR_30_OVERSAMPLING = "BALANCE_MENOR_30_OVERSAMPLING"
    BALANCE_MENOR_30_ENSMOTE = "BALANCE_MENOR_30_ENSMOTE"
    BALANCE_MENOR_30_ENSMOTE_TOMEK = "BALANCE_MENOR_30_ENSMOTE_TOMEK"
    NO_BALANCE = "NO_BALANCE"
    
class BalanceSolution(Enum):
    pass

class BalanceAnalyzer(AnalysisStep):
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> dict:
        pass