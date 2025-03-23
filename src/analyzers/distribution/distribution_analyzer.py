from enum import Enum
import pandas as pd
from scipy import stats
from src.analyzers.base.analysis_base import AnalysisStep



class DistributionAnalyzer(AnalysisStep):
    def __init__(self, min_samples: int = 20):
        self.min_samples = min_samples

    def analyze(self, data: pd.DataFrame) -> list:
        results = []
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()

        for col in numeric_cols:
            series = data[col].dropna()

            if set(series.unique()).issubset({0, 1}) or series.nunique() <= 1:
                continue
            if len(series) < self.min_samples:
                continue

            stats = self._calculate_statistics(series)
            dist_type = self._classify_distribution(stats)
            actions = self._get_suggested_actions(stats, dist_type)

            suggestion = actions[0] if actions else 'NENHUMA'
            result = {
                'column': col,
                'problem': 'distribution_pattern',
                'suggestion': suggestion,
                'actions': actions,
                'statistics': stats
            }
            results.append(result)

        return results

    def _calculate_statistics(self, series: pd.Series) -> dict:
        mean = series.mean()
        median = series.median()
        std = series.std()
        skewness = stats.skew(series, bias=False)
        kurt = stats.kurtosis(series, fisher=True)
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        try:
            sample = series.sample(min(1000, len(series)))
            shapiro_p = stats.shapiro(sample)[1]
            ks_p = stats.kstest(stats.zscore(sample), 'norm')[1]
        except Exception:
            shapiro_p = 0.0
            ks_p = 0.0

        return {
            'mean': mean,
            'median': median,
            'std': std,
            'skewness': skewness,
            'kurtosis': kurt,
            'shapiro_p': shapiro_p,
            'ks_p': ks_p,
            'q1': q1,
            'q3': q3,
            'iqr': iqr
        }

    def _classify_distribution(self, stats: dict):
        skewness = stats['skewness']
        kurtosis = stats['kurtosis']
        shapiro_p = stats['shapiro_p']

        is_normal = abs(skewness) < 0.5 and abs(kurtosis) < 0.5 or shapiro_p > 0.05
        if is_normal:
            return 'NORMAL'

        if abs(skewness) < 0.2 and kurtosis < -0.5:
            return 'UNIFORME'

        if skewness > 0.5:
            if skewness > 1.5 and kurtosis > 3:
                return 'LOGNORMAL'
            if abs(skewness - 2) < 0.5 and abs(kurtosis - 6) < 1.5:
                return 'EXPONENCIAL'
            return 'ASSIMETRICA_POSITIVA'

        if skewness < -0.5:
            return 'ASSIMETRICA_NEGATIVA'

        if kurtosis > 3:
            return 'CAUDA_PESADA'

        return 'DESCONHECIDA'

    def _get_suggested_actions(self, stats: dict, dist_type: str) -> list:
        skewness = stats['skewness']
        actions = []

        if dist_type == 'NORMAL':
            return []

        if dist_type == 'UNIFORME':
            return ['NENHUMA']

        if dist_type in ['ASSIMETRICA_POSITIVA', 'LOGNORMAL', 'EXPONENCIAL']:
            if skewness > 2:
                actions = ['TRANSFORMACAO_LOG', 'TRANSFORMACAO_BOX_COX']
            elif skewness > 1:
                actions = ['TRANSFORMACAO_LOG', 'TRANSFORMACAO_RAIZ_QUADRADA']
            else:
                actions = ['TRANSFORMACAO_RAIZ_QUADRADA', 'CENTRALIZACAO_PADRONIZACAO']

        elif dist_type == 'ASSIMETRICA_NEGATIVA':
            if skewness < -2:
                actions = ['TRANSFORMACAO_REFLECAO_LOG']
            else:
                actions = ['TRANSFORMACAO_QUADRATICA', 'TRANSFORMACAO_INVERSA']

        elif dist_type == 'CAUDA_PESADA':
            actions = ['WINSORIZATION', 'CLIPPING', 'ALGORITMO_ROBUSTO']

        return actions
