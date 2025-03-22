# agents/cross_validator.py
from typing import Dict, Any
import pandas as pd
from .base_agent import EDAAgent

class CrossValidator(EDAAgent):
    def analyze(self, data: pd.DataFrame, context: Dict[str, Any]) -> Dict[str, Any]:
        # Coletar resultados de todos os agentes anteriores
        missing_results = context.get('MissingValuesAnalyzer', {})
        outlier_results = context.get('OutlierAnalyzer', {})
        feature_scores = context.get('FeatureScorer', [])

        # Lista para armazenar inconsistências encontradas
        inconsistencies = []

        # Validação 1: Variáveis com muitos ausentes não devem ter alta pontuação
        for feature in feature_scores:
            col = feature['column']
            score = feature['score_total']
            missing_info = next((item for item in missing_results if item['column'] == col), None)
            
            if missing_info and missing_info['problem'] == 'MUITOS_AUSENTES' and score > 3:
                inconsistencies.append(f"⚠️ Conflito: {col} tem muitos ausentes mas alta pontuação ({score})")

        # Validação 2: Outliers em variáveis consideradas normais
        for outlier in outlier_results:
            col = outlier['column']
            dist_info = next((item for item in context.get('DistributionAnalyzer', []) if item['column'] == col), None)
            
            if dist_info and dist_info['problem'] == 'NORMAL' and outlier['problem'] == 'MUITOS_OUTLIERS':
                inconsistencies.append(f"⚠️ Conflito: {col} foi classificada como normal mas tem muitos outliers")

        return {
            'inconsistencies': inconsistencies,
            'passed': len(inconsistencies) == 0
        }