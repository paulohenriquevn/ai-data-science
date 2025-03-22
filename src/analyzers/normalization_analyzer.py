from enum import Enum
from typing import List, Tuple
from abc import abstractmethod
import pandas as pd
from src.analyzers.analysis_step import AnalysisStep

class NormalizationTechnique(Enum):
    """Técnicas de normalização conforme a tabela"""
    MIN_MAX_SCALING = "Min-Max Scaling"
    STANDARD_SCALING = "Standard Scaling (Z-score)"
    MAXABS_SCALING = "MaxAbs Scaling"
    ROBUST_SCALING = "Robust Scaling"
    L1_NORMALIZATION = "Normalização L1 (Manhattan)"
    L2_NORMALIZATION = "Normalização L2 (Euclidiana)"
    LOG_TRANSFORM = "Log Transform"
    BOX_COX = "Box-Cox Transform"
    YEO_JOHNSON = "Yeo-Johnson Transform"
    QUANTILE_TRANSFORM = "Quantile Transform"
    UNIT_VECTOR = "Unit Vector Transform"
    MEAN_NORMALIZATION = "Mean Normalization"
    SIGMOIDAL = "Sigmoidal/Logistic Transform"
    DECIMAL_SCALING = "Decimal Scaling"
    ADAPTIVE_SCALING = "Escalonamento Adaptativo"
    POWER_TRANSFORM = "Power Transform"

class NormalizationScenario(Enum):
    """Cenários de normalização seguindo o padrão type + description"""
    
    MIN_MAX_CASE = (
        NormalizationTechnique.MIN_MAX_SCALING, 
        "Quando os limites são conhecidos e significativos. Quando a interpretabilidade é importante. Quando zeros precisam ser preservados"
    )
    
    STANDARD_SCALING_CASE = (
        NormalizationTechnique.STANDARD_SCALING,
        "Para distribuições aproximadamente normais. Para métodos baseados em covariância. Quando os outliers não são extremos"
    )
    
    MAXABS_CASE = (
        NormalizationTechnique.MAXABS_SCALING,
        "Para dados esparsos. Quando preservação de zeros é crítica. Quando o sinal é importante"
    )
    
    ROBUST_SCALING_CASE = (
        NormalizationTechnique.ROBUST_SCALING,
        "Quando há outliers significativos. Dados com distribuições muito assimétricas. Quando a mediana é mais relevante"
    )
    
    L1_NORM_CASE = (
        NormalizationTechnique.L1_NORMALIZATION,
        "Para vetores de características esparsos. Em NLP e processamento de texto. Quando a esparsidade deve ser preservada"
    )
    
    L2_NORM_CASE = (
        NormalizationTechnique.L2_NORMALIZATION,
        "Para comparações de similaridade. Quando a direção dos vetores importa. Para embeddings e representações vetoriais"
    )
    
    LOG_TRANSFORM_CASE = (
        NormalizationTechnique.LOG_TRANSFORM,
        "Distribuições com assimetria positiva forte. Dados financeiros, populacionais. Variáveis com crescimento exponencial"
    )
    
    BOX_COX_CASE = (
        NormalizationTechnique.BOX_COX,
        "Quando a normalidade é importante. Para dados positivos com assimetria. Em análises estatísticas formais"
    )
    
    YEO_JOHNSON_CASE = (
        NormalizationTechnique.YEO_JOHNSON,
        "Quando há valores negativos. Distribuições assimétricas. Quando Box-Cox não é aplicável"
    )
    
    QUANTILE_CASE = (
        NormalizationTechnique.QUANTILE_TRANSFORM,
        "Quando a forma da distribuição é mais importante que os valores. Para dados muito assimétricos ou com outliers"
    )
    
    UNIT_VECTOR_CASE = (
        NormalizationTechnique.UNIT_VECTOR,
        "Quando apenas a direção do vetor importa. Para comparações de similaridade. Análise de texto e documentos"
    )
    
    MEAN_NORM_CASE = (
        NormalizationTechnique.MEAN_NORMALIZATION,
        "Quando tanto centralização quanto escala são importantes. Alternativa ao z-score quando a amplitude é mais relevante"
    )
    
    SIGMOIDAL_CASE = (
        NormalizationTechnique.SIGMOIDAL,
        "Para conversão em probabilidades. Como passo de ativação em redes neurais. Quando valores limitados são desejados"
    )
    
    DECIMAL_SCALING_CASE = (
        NormalizationTechnique.DECIMAL_SCALING,
        "Para fins de visualização. Análise preliminar. Quando a fácil interpretação é crucial"
    )
    
    ADAPTIVE_SCALING_CASE = (
        NormalizationTechnique.ADAPTIVE_SCALING,
        "Datasets heterogêneos. Quando performance é crítica. Com recursos computacionais disponíveis"
    )
    
    POWER_TRANSFORM_CASE = (
        NormalizationTechnique.POWER_TRANSFORM,
        "Dados com assimetria moderada. Quando relações de potência são esperadas. Para estabilizar variância"
    )

    def __init__(self, technique: NormalizationTechnique, description: str):
        self.technique = technique
        self.description = description
class NormalizationAnalyzer(AnalysisStep):
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> dict:
        pass