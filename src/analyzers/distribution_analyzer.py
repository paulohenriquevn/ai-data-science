from enum import Enum
from typing import List
from abc import abstractmethod
import pandas as pd
from src.analyzers.analysis_step import AnalysisStep

class DistributionType(Enum):
    """Tipos de distribuições conforme a tabela"""
    NORMAL = "Normal (Gaussiana)"
    ASSIMETRICA_POSITIVA = "Assimétrica positiva (direita)"
    ASSIMETRICA_NEGATIVA = "Assimétrica negativa (esquerda)"
    UNIFORME = "Uniforme"
    BIMODAL = "Bimodal"
    MULTIMODAL = "Multimodal"
    LOGNORMAL = "Lognormal"
    EXPONENCIAL = "Exponencial"
    POISSON = "Poisson"
    BINOMIAL = "Binomial"
    BINOMIAL_NEGATIVA = "Binomial Negativa"
    GAMMA = "Gamma"
    WEIBULL = "Weibull"
    BETA = "Beta"
    CAUDA_PESADA = "Distribuições de cauda pesada"
    TRUNCADA = "Distribuições limitadas (truncadas)"
    MISTA = "Distribuições mistas"

class DistributionSolution(Enum):
    """Técnicas de transformação e modelos da tabela"""
    # Técnicas de Transformação
    TRANSFORMACAO_LOG = "Log"
    TRANSFORMACAO_RAIZ_QUADRADA = "Raiz quadrada"
    TRANSFORMACAO_BOX_COX = "Box-Cox"
    TRANSFORMACAO_QUADRATICA = "Transformação quadrática"
    TRANSFORMACAO_REFLECAO_LOG = "Reflexão + log"
    WINSORIZACAO = "Winsorização"
    RANK_TRANSFORM = "Rank-transformation"
    TRANSFORMACAO_LOGIT = "Transformação logit"
    
    # Modelos
    REGRESSAO_LINEAR = "Regressão linear"
    ANOVA = "ANOVA"
    TESTES_T_Z = "Testes t e z"
    LDA = "LDA"
    GLM_GAMMA = "GLM com distribuição gamma"
    MODELOS_NAO_PARAMETRICOS = "Modelos não-paramétricos"
    MODELOS_MISTURA = "Modelos de mistura"
    GLM_LOG = "GLM com link log"
    MODELOS_SOBREVIVENCIA = "Modelos de sobrevivência"
    REGRESSAO_POISSON = "Regressão de Poisson"
    REGRESSAO_LOGISTICA = "Regressão logística"
    REGRESSAO_QUANTILICA = "Regressão quantílica"
    MODELOS_TOBIT = "Modelos Tobit"

class DistributionScenario(Enum):
    """Associação completa de cada distribuição com características e soluções"""
    
    NORMAL = (
        [],  # Sem transformação necessária
        [DistributionSolution.REGRESSAO_LINEAR, DistributionSolution.ANOVA, DistributionSolution.TESTES_T_Z],
        "Simétrica, média = mediana = moda. Base para métodos paramétricos"
    )
    
    ASSIMETRICA_POSITIVA = (
        [DistributionSolution.TRANSFORMACAO_LOG, DistributionSolution.TRANSFORMACAO_RAIZ_QUADRADA, 
        DistributionSolution.TRANSFORMACAO_BOX_COX],
        [DistributionSolution.GLM_GAMMA, DistributionSolution.MODELOS_NAO_PARAMETRICOS],
        "Cauda direita alongada. Comum em dados financeiros e tempo de espera"
    )
    
    LOGNORMAL = (
        [DistributionSolution.TRANSFORMACAO_LOG],
        [DistributionSolution.GLM_LOG, DistributionSolution.MODELOS_MULTIPLICATIVOS],
        "Log dos dados segue distribuição normal. Comum em biologia e finanças"
    )
    
    # Padrão continua para todas as distribuições...
    
    def __init__(self, 
                transformacoes: List[DistributionSolution],
                modelos: List[DistributionSolution],
                descricao: str):
        self.transformacoes = transformacoes
        self.modelos = modelos
        self.descricao = descricao

class DistributionAnalyzer(AnalysisStep):
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> dict:
        pass