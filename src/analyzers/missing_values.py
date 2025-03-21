from enum import Enum
import pandas as pd
from src.analyzers.analysis_step import AnalysisStep


class MissingValuesProblem(Enum):
    LESS_5 = "LESS_5"
    BETWEEN_5_30 = "BETWEEN_5_30"
    GREATER_30 = "GREATER_30"
    NO_MISSING_VALUES = "NO_MISSING_VALUES"

    
class MissingValuesSolution(Enum):
    IMPUTATION_MEDIA = "IMPUTATION_MEDIA"
    IMPUTATION_MEDIANA = "IMPUTATION_MEDIANA"
    REMOVE_COLUMN = "REMOVE_COLUMN"


class MissingValuesAnalyzer(AnalysisStep):
    def __init__(self):
        self.result = []

    def analyze(self, data: pd.DataFrame) -> dict:
        self.result = []
        for col in data.columns:
            if data[col].isnull().sum() > 0:
                missing_ratio = data[col].isnull().mean()
                if missing_ratio < 0.05:
                    solution = MissingValuesSolution.IMPUTATION_MEDIA
                    problem_enum = MissingValuesProblem.LESS_5
                elif missing_ratio <= 0.3:
                    solution = MissingValuesSolution.IMPUTATION_MEDIANA
                    problem_enum = MissingValuesProblem.BETWEEN_5_30
                else:
                    solution = MissingValuesSolution.REMOVE_COLUMN
                    problem_enum = MissingValuesProblem.GREATER_30

                self.result.append({
                    'column': col,
                    'problem': problem_enum.name,
                    'solution': solution
                })

        if not self.result:
            self.result = [{
                'column': [],
                'problem': MissingValuesProblem.NO_MISSING_VALUES.name,
                'solution': None
            }]
        return self.result