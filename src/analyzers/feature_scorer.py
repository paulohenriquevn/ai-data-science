from typing import List, Dict, Any
from collections import defaultdict
import pandas as pd
from src.analyzers.base.analysis_base import AnalysisStep
from src.utils import detect_and_replace_placeholders

class FeatureScorer(AnalysisStep):
    """
    Consolida os resultados das análises anteriores e gera uma pontuação para cada variável.
    Considera: correlação, significância estatística, outliers, distribuição e valores ausentes.
    """

    def __init__(self):
        # pesos de cada critério
        self.pesos = {
            'CORRELACAO_COM_TARGET': 3,
            'CORRELACAO_FORTE_POSITIVA': 4,
            'CORRELACAO_FORTE_NEGATIVA': 4,
            'CORRELACAO_MODERADA': 2,
            'VARIAVEL_SIGNIFICATIVA': 3,
            'MUITOS_OUTLIERS': -1,
            'DISTRIBUICAO_NAO_NORMAL': 1,
            'DISTRIBUICAO_NORMAL': 2,
            'POUCOS_AUSENTES': 1,
            'MUITOS_AUSENTES': -3,
            'TARGET_AUSENTE': -4,
            'CATEGORICOS': 1,
            'RELACIONADA': 2
        }

    def analyze(self, data: pd.DataFrame, analises: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        data = detect_and_replace_placeholders(data)
        score_map = defaultdict(lambda: {'score': 0, 'justificativas': []})

        for resultado_analise in analises:
            for item in resultado_analise:
                col = item['column']
                problema = item['problem']
                peso = self.pesos.get(problema, 0)
                score_map[col]['score'] += peso
                score_map[col]['justificativas'].append(f"{problema} (peso={peso})")

        resultados = []
        for col, info in score_map.items():
            resultados.append({
                'column': col,
                'score_total': info['score'],
                'selecionar': info['score'] >= 3,
                'justificativas': info['justificativas']
            })

        return sorted(resultados, key=lambda x: x['score_total'], reverse=True)
