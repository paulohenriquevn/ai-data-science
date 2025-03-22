from enum import Enum
from typing import List
from abc import abstractmethod
import pandas as pd
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from src.analyzers.analysis_step import AnalysisStep
from src.utils import detect_and_replace_placeholders

class OutliersProblem(Enum):
    """Tipos de problemas de outliers conforme a tabela"""
    OUTLIERS_NORMAIS = "Outliers estatísticos em distribuição normal"
    OUTLIERS_NAO_NORMAIS = "Outliers em distribuições não-normais"
    SERIES_TEMPORAIS = "Outliers em séries temporais"
    MULTIVARIADOS = "Outliers multivariados"
    ALTA_DIMENSIONALIDADE = "Outliers em alta dimensionalidade"
    CATEGORICOS = "Outliers em dados categóricos"
    ERROS_MEDICAO = "Erros de medição/entrada"
    OUTLIERS_GENUINOS = "Outliers genuínos (informativos)"
    REGRESSAO = "Outliers em modelos de regressão"
    CLASSIFICACAO = "Outliers em modelos de classificação"
    CLUSTERING = "Outliers em clustering"
    TEMPORAIS_PERSISTENTES = "Outliers temporários vs. persistentes"
    DADOS_ESPARSO = "Outliers em dados esparsos"
    GRUPOS_DESBALANCEADOS = "Outliers em grupos desbalanceados"

class OutliersSolution(Enum):
    """Técnicas de tratamento de outliers da tabela"""
    # Técnicas Gerais
    REMOCAO = "Remoção"
    TRANSFORMACAO_LOG = "Transformação logarítmica"
    WINSORIZACAO = "Winsorização"
    CATEGORIZACAO = "Categorização em buckets"
    MODELAGEM_ROBUSTA = "Modelagem robusta"
    IMPUTACAO_CONTEXTUAL = "Imputação contextual"
    ANALISE_INFLUENCIA = "Análise de influência"
    
    # Técnicas Específicas
    REGRESSAO_QUANTILICA = "Regressão quantílica"
    ENSEMBLE_TECNICAS = "Técnicas de ensemble"
    CLUSTERING_REJEICAO = "Clustering com rejeição"
    TRATAMENTO_ESPECIFICO_GRUPO = "Tratamento específico por grupo"
    SUAVIZACAO_MEDIA_MOVEL = "Suavização (média móvel)"
    REDUCAO_DIMENSIONALIDADE = "Redução de dimensionalidade + detecção"
    AUTOENCODER = "Autoencoder"
    REGULARIZACAO_ESPECIFICA = "Regularização específica"

class OutliersScenario(Enum):
    """Associação de cenários com soluções e descrições"""
    
    OUTLIERS_NORMAIS = (
        [OutliersSolution.REMOCAO, OutliersSolution.TRANSFORMACAO_LOG, OutliersSolution.WINSORIZACAO],
        "Efetivo apenas para distribuições aproximadamente normais",
        "Z-score (±3), Regra do IQR (1,5 × IQR)"
    )
    
    OUTLIERS_NAO_NORMAIS = (
        [OutliersSolution.WINSORIZACAO, OutliersSolution.CATEGORIZACAO, OutliersSolution.TRANSFORMACAO_LOG],
        "Considerar a assimetria da distribuição",
        "IQR modificado, MAD (Desvio Absoluto Mediano)"
    )
    
    SERIES_TEMPORAIS = (
        [OutliersSolution.SUAVIZACAO_MEDIA_MOVEL, OutliersSolution.IMPUTACAO_CONTEXTUAL],
        "Considerar sazonalidade e tendências",
        "Decomposição de série temporal, Filtro de Kalman"
    )
    
    MULTIVARIADOS = (
        [OutliersSolution.REMOCAO, OutliersSolution.ANALISE_INFLUENCIA, OutliersSolution.MODELAGEM_ROBUSTA],
        "Pode ser normal em uma variável, mas anômalo no contexto multivariado",
        "Distância de Mahalanobis, Isolation Forest"
    )
    
    ALTA_DIMENSIONALIDADE = (
        [OutliersSolution.REDUCAO_DIMENSIONALIDADE, OutliersSolution.AUTOENCODER],
        "Maldição da dimensionalidade afeta a detecção",
        "LOF, HDBSCAN, Isolation Forest"
    )
    
    CATEGORICOS = (
        [OutliersSolution.CATEGORIZACAO, OutliersSolution.MODELAGEM_ROBUSTA],
        "Considerar o significado semântico da raridade",
        "Frequência de categorias, Análise de associação"
    )
    
    ERROS_MEDICAO = (
        [OutliersSolution.REMOCAO, OutliersSolution.IMPUTACAO_CONTEXTUAL],
        "Importante diferenciar de outliers genuínos",
        "Validação de domínio, Verificação de limite físico"
    )
    
    OUTLIERS_GENUINOS = (
        [OutliersSolution.MODELAGEM_ROBUSTA, OutliersSolution.ANALISE_INFLUENCIA],
        "Podem conter informações valiosas",
        "Validação com especialistas, Análise contextual"
    )
    
    REGRESSAO = (
        [OutliersSolution.REGRESSAO_QUANTILICA, OutliersSolution.MODELAGEM_ROBUSTA],
        "Diferenciar entre outliers em X e outliers em Y",
        "Distância de Cook, Leverage (hat values)"
    )
    
    CLASSIFICACAO = (
        [OutliersSolution.ENSEMBLE_TECNICAS, OutliersSolution.MODELAGEM_ROBUSTA],
        "Podem afetar diretamente a fronteira de decisão",
        "Cross-validation com análise de erro, Métodos baseados em ensemble"
    )
    
    CLUSTERING = (
        [OutliersSolution.CLUSTERING_REJEICAO, OutliersSolution.MODELAGEM_ROBUSTA],
        "Podem formar clusters próprios ou distorcer clusters existentes",
        "Silhouette score, DBSCAN, HDBSCAN"
    )
    
    TEMPORAIS_PERSISTENTES = (
        [OutliersSolution.TRATAMENTO_ESPECIFICO_GRUPO, OutliersSolution.MODELAGEM_ROBUSTA],
        "Importante diferenciar anomalias temporárias de mudanças estruturais",
        "Análise de janela deslizante, Métodos baseados em persistência"
    )
    
    DADOS_ESPARSO = (
        [OutliersSolution.REGULARIZACAO_ESPECIFICA, OutliersSolution.MODELAGEM_ROBUSTA],
        "A esparsidade dificulta a detecção convencional",
        "Algoritmos específicos para dados esparsos"
    )
    
    GRUPOS_DESBALANCEADOS = (
        [OutliersSolution.TRATAMENTO_ESPECIFICO_GRUPO, OutliersSolution.MODELAGEM_ROBUSTA],
        "O que é outlier em um grupo pode ser normal em outro",
        "Estratificação + detecção, Análise de propensão"
    )

    def __init__(self, 
                solucoes: List[OutliersSolution],
                observacoes: str,
                tecnicas_deteccao: str):
        self.solucoes = solucoes
        self.observacoes = observacoes
        self.tecnicas_deteccao = tecnicas_deteccao



class OutlierAnalyzer(AnalysisStep):
    """
    Detecta outliers em variáveis numéricas com base no método do IQR (Interquartil).
    Também calcula a proporção de outliers por variável.
    """

    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold  # proporção mínima de outliers para considerar relevante

    def analyze(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        results = []
        df = detect_and_replace_placeholders(data)
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            serie = df[col].dropna()
            if len(serie) < 10:
                continue

            q1 = serie.quantile(0.25)
            q3 = serie.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            outliers = serie[(serie < lower_bound) | (serie > upper_bound)]
            outlier_ratio = len(outliers) / len(serie)

            if outlier_ratio >= self.threshold:
                results.append({
                    'column': col,
                    'problem': 'MUITOS_OUTLIERS',
                    'problem_description': 'Alta proporção de outliers detectada com base no IQR',
                    'description': 'Variável com muitos valores extremos',
                    'solution': 'TRATAMENTO_OUTLIERS',
                    'actions': ['WINSORIZACAO', 'TRANSFORMACAO_LOG', 'REMOVER_OUTLIERS'],
                    'statistics': {
                        'q1': round(q1, 3),
                        'q3': round(q3, 3),
                        'iqr': round(iqr, 3),
                        'outlier_ratio': round(outlier_ratio, 4),
                        'num_outliers': int(len(outliers))
                    }
                })

        return results
