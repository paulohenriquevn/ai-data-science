from enum import Enum
from typing import List
from abc import abstractmethod
import pandas as pd
from src.analyzers.analysis_step import AnalysisStep

class CorruptionType(Enum):
    """Tipos de problemas de dados corrompidos/inconsistentes da tabela"""
    VALORES_IMPOSSIVEIS = "Valores Impossíveis"
    INCONSISTENCIAS_ENTRE_VARIAVEIS = "Inconsistências entre Variáveis"
    ERROS_DIGITACAO = "Erros de Digitação"
    INCONSISTENCIAS_FORMATO = "Inconsistências de Formato"
    DUPLICATAS = "Duplicatas"
    INCONSISTENCIAS_TEMPORAIS = "Inconsistências Temporais"
    VALORES_ATIPICOS = "Valores Atípicos (Outliers)"
    CODIFICACAO_INCORRETA = "Codificação Incorreta"
    VALORES_TRUNCADOS = "Valores Truncados ou Incompletos"
    INCONSISTENCIAS_FONTES = "Inconsistências entre Fontes"
    PROBLEMAS_ESCALA = "Problemas de Escala/Unidade"
    CENSURA_DADOS = "Censura de Dados"
    DADOS_MASCARADOS = "Valores Mascarados (Privacy)"
    CORRUPCAO_ARMAZENAMENTO = "Dados Corrompidos em Armazenamento"
    INCONSISTENCIAS_SEMANTICAS = "Inconsistências Semânticas"
    REGISTROS_FANTASMAS = "Registros Fantasmas/Zumbis"
    ERROS_CALCULO = "Erros de Cálculo Persistentes"

class CorruptionSolution(Enum):
    """Técnicas de tratamento mapeadas da coluna 'Técnicas de Tratamento'"""
    # Valores Impossíveis
    SUBSTITUICAO_VALORES_PLAUSIVEIS = "Substituição por valores plausíveis"
    IMPUTACAO_ESTATISTICA = "Imputação estatística"
    FLAGGING_EXCLUSAO = "Flagging e exclusão"
    CORRECAO_REGRA_NEGOCIO = "Correção baseada em regras de negócio"
    
    # Inconsistências entre Variáveis
    REVISAO_CONJUNTA = "Revisão conjunta de variáveis"
    RECONCILIACAO_REGRA = "Reconciliação baseada em regras"
    PRIORIZACAO_CONFIABILIDADE = "Priorização por confiabilidade"
    EXCLUSAO_AMBIGUIDADE = "Exclusão em caso de ambiguidade sem solução"
    
    # Erros de Digitação
    CORRECAO_AUTOMATIZADA = "Correção automatizada para casos simples"
    FUZZY_MATCHING = "Fuzzy matching"
    PADRONIZACAO_MANUAL = "Padronização manual para casos críticos"
    DICIONARIO_CORRECAO = "Dicionários de correção"
    
    # Demais técnicas continuam seguindo o mesmo padrão...
    # ... (implementar todas as 70+ técnicas da tabela)

class CorruptionScenario(Enum):
    """Associação de cada tipo de problema com suas soluções e descrições"""
    
    VALORES_IMPOSSIVEIS = (
        [
            CorruptionSolution.SUBSTITUICAO_VALORES_PLAUSIVEIS,
            CorruptionSolution.IMPUTACAO_ESTATISTICA,
            CorruptionSolution.FLAGGING_EXCLUSAO,
            CorruptionSolution.CORRECAO_REGRA_NEGOCIO
        ],
        "Valores fora dos limites físicos ou lógicos aceitáveis. Distorcem estatísticas e previsões."
    )
    
    INCONSISTENCIAS_ENTRE_VARIAVEIS = (
        [
            CorruptionSolution.REVISAO_CONJUNTA,
            CorruptionSolution.RECONCILIACAO_REGRA,
            CorruptionSolution.PRIORIZACAO_CONFIABILIDADE,
            CorruptionSolution.EXCLUSAO_AMBIGUIDADE
        ],
        "Conflitos entre variáveis relacionadas. Causam resultados contraditórios e baixa performance."
    )
    
    ERROS_DIGITACAO = (
        [
            CorruptionSolution.CORRECAO_AUTOMATIZADA,
            CorruptionSolution.FUZZY_MATCHING,
            CorruptionSolution.PADRONIZACAO_MANUAL,
            CorruptionSolution.DICIONARIO_CORRECAO
        ],
        "Erros de entrada de dados que fragmentam categorias e prejudicam análises."
    )
    
    # Padrão continua para todos os 17 tipos de problemas...
    
    def __init__(self, solutions: List[CorruptionSolution], description: str):
        self.solutions = solutions
        self.description = description

class CorruptionAnalyzer(AnalysisStep):
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> dict:
        pass