import pandas as pd

class OutlierAnalyzer:
    def __init__(self, skew_threshold: float = 1.0, outlier_threshold: float = 0.1):
        self.skew_threshold = skew_threshold
        self.outlier_threshold = outlier_threshold

    def analyze(self, df: pd.DataFrame) -> list:
        results = []
        numeric_cols = df.select_dtypes(include='number').columns

        for col in numeric_cols:
            series = df[col].dropna()

            if series.nunique() <= 1:
                continue  # ignora colunas constantes

            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = series[(series < lower_bound) | (series > upper_bound)]
            outlier_count = outliers.count()
            outlier_ratio = outlier_count / len(series)

            skewness = series.skew()

            actions = self._suggest_actions(skewness, outlier_ratio)
            suggestion = actions[0] if actions else 'NENHUMA'

            if outlier_ratio == 0:
                problem_description = f'Nenhum outlier detectado'
            else:
                problem_description = f'Distribuição {"assimétrica" if abs(skewness) > 0.5 else "simétrica"} e {outlier_ratio:.1%} de outliers detectados'

            results.append({
                'column': col,
                'problem': 'outlier_detection',
                'problem_description': problem_description,
                'suggestion': suggestion,
                'actions': actions,
                'statistics': {
                    'q1': q1,
                    'q3': q3,
                    'iqr': iqr,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound,
                    'outlier_count': outlier_count,
                    'outlier_ratio': outlier_ratio,
                    'skewness': skewness
                }
            })

        return results


    def compare(self, original_df: pd.DataFrame, transformed_df: pd.DataFrame, columns: list) -> pd.DataFrame:
        resultados = []

        for col in columns:
            original = original_df[col].dropna()
            novo = transformed_df[col].dropna()

            if original.empty or novo.empty:
                continue

            def get_stats(series):
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outliers = ((series < lower) | (series > upper)).sum()
                return {
                    "skewness": series.skew(),
                    "kurtosis": series.kurtosis(),
                    "outlier_ratio": outliers / len(series)
                }

            stats_before = get_stats(original)
            stats_after = get_stats(novo)

            for metric in stats_before:
                before = stats_before[metric]
                after = stats_after[metric]
                delta = ((after - before) / abs(before)) * 100 if before != 0 else 0
                resultados.append({
                    "variavel": col,
                    "metrica": metric,
                    "antes": before,
                    "depois": after,
                    "variacao_percentual": delta
                })

        return pd.DataFrame(resultados)


    """
    Suggest actions based on skewness and outlier ratio.
    """
    def _suggest_actions(self, skewness, outlier_ratio) -> list:
        actions = []

        if outlier_ratio <= 0.05 and abs(skewness) < 0.5:
            actions = ['NENHUMA']
        elif outlier_ratio > 0.1 or abs(skewness) > self.skew_threshold:
            if skewness > 1:
                actions = ['WINSORIZATION', 'TRANSFORMACAO_LOG', 'REMOCAO_OUTLIERS']
            elif skewness < -1:
                actions = ['WINSORIZATION', 'TRANSFORMACAO_REFLEXAO_LOG', 'REMOCAO_OUTLIERS']
            else:
                actions = ['WINSORIZATION', 'CLIPPING', 'REMOCAO_OUTLIERS']
        else:
            actions = ['CLIPPING']

        return actions

