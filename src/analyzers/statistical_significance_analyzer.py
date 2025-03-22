from typing import List, Dict, Any
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu
from src.analyzers.analysis_step import AnalysisStep
from src.utils import detect_and_replace_placeholders

class StatisticalSignificanceAnalyzer(AnalysisStep):
    """
    Verifica a significância estatística das variáveis numéricas em relação ao alvo (y).
    Aplica t-test ou Mann-Whitney U Test conforme necessário.
    """

    def __init__(self, target: str = 'y', pvalue_threshold: float = 0.05):
        self.target = target
        self.pvalue_threshold = pvalue_threshold

    def analyze(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        results = []
        df = detect_and_replace_placeholders(data)
        features = [col for col in df.select_dtypes(include=[np.number]).columns if col != self.target]

        for col in features:
            grupo_0 = df[df[self.target] == 0][col].dropna()
            grupo_1 = df[df[self.target] == 1][col].dropna()

            if len(grupo_0) < 20 or len(grupo_1) < 20:
                continue

            try:
                stat, p = ttest_ind(grupo_0, grupo_1, equal_var=False, nan_policy='omit')
                test_type = 't-test'
            except Exception:
                stat, p = mannwhitneyu(grupo_0, grupo_1, alternative='two-sided')
                test_type = 'mannwhitney'

            significant = p < self.pvalue_threshold

            results.append({
                'column': col,
                'problem': 'VARIAVEL_SIGNIFICATIVA' if significant else 'VARIAVEL_NAO_SIGNIFICATIVA',
                'problem_description': 'Diferença estatística entre classes de y',
                'description': f'Resultado do {test_type} entre y=0 e y=1',
                'solution': 'PRIORIZAR_USO_MODELO' if significant else 'MONITORAR_IMPORTANCIA',
                'actions': ['SELECAO_VARIAVEIS'] if significant else ['AVALIAR_RELEVANCIA'],
                'statistics': {
                    'test': test_type,
                    'statistic': round(stat, 3),
                    'p_value': round(p, 5),
                    'significant': significant
                }
            })

        return results
