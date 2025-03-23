import pandas as pd


class LowVariabilityAnalyzer:
    def __init__(self, dominance_threshold: float = 0.99):
        self.dominance_threshold = dominance_threshold

    def analyze(self, df: pd.DataFrame) -> list:
        results = []

        for col in df.columns:
            series = df[col].dropna()
            if series.nunique() <= 1:
                result = self._build_result(col, series, 1.0)
                results.append(result)
            else:
                most_freq_pct = series.value_counts(normalize=True).iloc[0]
                if most_freq_pct >= self.dominance_threshold:
                    result = self._build_result(col, series, most_freq_pct)
                    results.append(result)

        return results

    def _build_result(self, col, series, most_freq_pct) -> dict:
        return {
            'column': col,
            'problem': 'low_variability',
            'problem_description': f'A coluna possui {"apenas um valor distinto" if series.nunique() == 1 else f"{most_freq_pct:.1%} do mesmo valor"}.',
            'suggestion': 'REMOVER_CONSTANTE',
            'actions': ['REMOVER_CONSTANTE'],
            'statistics': {
                'unique_values': series.nunique(),
                'most_frequent_percentage': most_freq_pct
            }
        }
