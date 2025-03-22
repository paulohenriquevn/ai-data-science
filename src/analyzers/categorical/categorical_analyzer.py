import pandas as pd
from src.analyzers.base.analysis_base import AnalysisStep

class CategoricalAnalyzer(AnalysisStep):
    def __init__(self, max_unique_values=20):
        self.max_unique_values = max_unique_values

    def analyze(self, df: pd.DataFrame) -> list[dict]:
        report = []
        for col in df.columns:
            if df[col].dtype == 'object':
                report.append({"column": col, "strategy": "ONE_HOT"})
            elif pd.api.types.is_categorical_dtype(df[col]):
                report.append({"column": col, "strategy": "ONE_HOT"})
            elif pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() <= self.max_unique_values:
                report.append({"column": col, "strategy": "ORDINAL"})
        return report
