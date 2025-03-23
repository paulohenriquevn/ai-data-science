import pandas as pd
from src.analyzers.base.analysis_base import AnalysisStep

class NormalizationAnalyzer(AnalysisStep):
    def __init__(self, skew_threshold: float = 0.5):
        self.skew_threshold = skew_threshold

    def analyze(self, df: pd.DataFrame) -> list:
        results = []
        numeric_cols = df.select_dtypes(include='number').columns

        for col in numeric_cols:
            series = df[col].dropna()

            if series.nunique() <= 1:
                continue

            mean = series.mean()
            std = series.std()
            skew = series.skew()
            min_val = series.min()
            max_val = series.max()

            if abs(skew) >= self.skew_threshold:
                suggestion = 'ROBUST_SCALER'
                actions = ['ROBUST_SCALER', 'STANDARD_SCALER']
            else:
                suggestion = 'STANDARD_SCALER'
                actions = ['STANDARD_SCALER', 'ROBUST_SCALER']

            results.append({
                'column': col,
                'problem': 'scaling_needed',
                'problem_description': f'Distribuição com skew {skew:.2f} e valores entre {min_val:.1f} e {max_val:.1f}',
                'suggestion': suggestion,
                'actions': actions,
                'statistics': {
                    'mean': mean,
                    'std': std,
                    'skewness': skew,
                    'min': min_val,
                    'max': max_val
                }
            })

        return results

