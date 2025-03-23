import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import skew, kurtosis

class MissingValuesAnalyzer:
    def __init__(self, target_column=None):
        self.target_column = target_column

    def analyze(self, data: pd.DataFrame) -> list:
        results = []
        missing_stats = self._calculate_missing_stats(data)
        
        for col in data.columns:
            result = self._analyze_column(data, col, missing_stats[col])
            results.append(result)
            
        return results

    def _analyze_column(self, data: pd.DataFrame, col: str, stats: dict) -> dict:
        missing_percent = stats['missing_percent']
        
        if missing_percent == 0:
            category = "Nenhum problema"
            actions = ['NENHUMA_ACAO_NECESSARIA']
            suggestion = 'NENHUMA_ACAO_NECESSARIA'
        elif missing_percent <= 5:
            category = "Poucos valores ausentes (≤5%)"
            actions = ['IMPUTE_MEDIA', 'IMPUTE_MEDIANA', 'EXCLUIR_LINHAS']
            suggestion = 'IMPUTE_MEDIA'
        elif missing_percent <= 30:
            category = "Valores ausentes moderados (>5% e ≤30%)"
            actions = ['ADICIONAR_FLAG_E_IMPUTAR', 'IMPUTE_MEDIANA', 'KNN_IMPUTER']
            suggestion = 'ADICIONAR_FLAG_E_IMPUTAR'
        elif missing_percent <= 60:
            category = "Muitos valores ausentes (>30% e ≤60%)"
            actions = ['MICE_IMPUTER', 'ADICIONAR_FLAG_E_IMPUTAR', 'EXCLUIR_COLUNA']
            suggestion = 'MICE_IMPUTER'
        else:
            category = "Valores ausentes críticos (>60%)"
            actions = ['EXCLUIR_COLUNA']
            suggestion = 'EXCLUIR_COLUNA'

        return {
            'column': col,
            'problem': 'missing_values',
            'problem_description': category,
            'suggestion': suggestion,
            'actions': actions,
            'statistics': stats
        }

    def _calculate_missing_stats(self, data: pd.DataFrame) -> dict:
        stats = {}
        for col in data.columns:
            missing_count = data[col].isna().sum()
            total_count = len(data[col])
            missing_percent = (missing_count / total_count) * 100 if total_count > 0 else 0

            stats[col] = {
                'missing_count': missing_count,
                'total_count': total_count,
                'missing_percent': missing_percent
            }
        return stats


