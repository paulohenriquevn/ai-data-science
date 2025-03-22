from enum import Enum
from typing import List
from abc import abstractmethod
import pandas as pd
from src.analyzers.analysis_step import AnalysisStep

class DuplicatesType(Enum):
    """Tipos de duplicatas conforme a tabela"""
    EXATAS = "Duplicatas Exatas"
    PARCIAIS = "Duplicatas Parciais"
    FUZZY = "Duplicatas Fuzzy (Aproximadas)"
    MULTI_FONTE = "Duplicatas Multi-Fonte"
    VALORES_DIFERENTES = "Duplicatas com Valores Diferentes"
    HIERARQUICAS = "Duplicatas Hierárquicas"
    TEMPORAIS = "Duplicatas Temporais"
    EVENTOS = "Duplicatas de Eventos"
    ESTRUTURADOS = "Duplicatas em Dados Estruturados"
    SEMI_ESTRUTURADOS = "Duplicatas em Dados Semi-estruturados"
    TRANSACIONAIS = "Duplicatas Transacionais"
    MIGRACAO = "Duplicatas em Migração de Dados"
    USUARIOS = "Duplicatas de Usuários/Entidades"
    BIG_DATA = "Duplicatas em Big Data"
    TEXTUAIS = "Duplicatas em Dados Textuais"
    ML = "Duplicatas em Machine Learning"

class DuplicatesSolution(Enum):
    """Todas as técnicas de tratamento da coluna 'Estratégias de Tratamento' mapeadas"""
    
    # Duplicatas Exatas
    REMOCAO_DUPLICATAS = "Remoção de duplicatas (drop_duplicates)"
    MANUTENCAO_PRIMEIRO_ULTIMO = "Manutenção do primeiro/último registro"
    AGREGACAO_POR_ID = "Agregação por ID"
    FLAGS_DUPLICACAO = "Criação de flags de duplicação"
    
    # Duplicatas Parciais
    PRIORIZACAO_REGISTROS_COMPLETOS = "Priorização de registros completos"
    MESCLAGEM_INFORMACOES = "Mesclagem de informações complementares"
    RESOLUCAO_BASEADA_REGRA = "Resolução baseada em regras"
    EXCLUSAO_AMBIGUIDADE = "Exclusão em caso de ambiguidade sem solução"
    
    # Duplicatas Fuzzy
    CLUSTERING_SIMILARIDADE = "Clustering por similaridade"
    CORRECAO_MANUAL_ASSISTIDA = "Correção manual assistida"
    CONSOLIDACAO_LIMIAR = "Consolidação com limiar"
    DEDUPLICACAO_ITERATIVA = "Deduplicação iterativa"
    
    # Duplicatas Multi-Fonte
    SISTEMA_ENTIDADE_MESTRA = "Sistema de entidade mestra (MDM)"
    HIERARQUIA_FONTES = "Hierarquia de fontes"
    TIMESTAMPS_ATUALIZACAO = "Timestamps de atualização"
    CONSOLIDACAO_REGRA_NEGOCIO = "Consolidação por regras de negócio"
    
    # Duplicatas com Valores Diferentes
    RESOLUCAO_REGRA_CONFLITO = "Resolução baseada em regras"
    REGISTRO_MAIS_RECENTE = "Escolha do registro mais recente"
    CONSOLIDACAO_COMPLETUDE = "Consolidação por completude"
    CRIACAO_VERSAO = "Criação de versões"
    
    # Duplicatas Hierárquicas
    NORMALIZACAO_HIERARQUIA = "Normalização hierárquica"
    CONSOLIDACAO_NIVEIS = "Consolidação em níveis"
    RESOLUCAO_TOP_DOWN = "Resolução top-down"
    RESTRUTURACAO_ARVORES = "Restruturação de árvores"
    
    # Duplicatas Temporais
    AGREGACAO_TEMPORAL = "Agregação temporal"
    REMOCAO_JANELA = "Remoção por janela"
    AMOSTRAGEM_CONTROLADA = "Amostragem controlada"
    CONSOLIDACAO_INTERVALO = "Consolidação por intervalo"
    
    # Duplicatas de Eventos
    DEDUPLICACAO_JANELA = "Deduplicação por janela"
    CONSOLIDACAO_EVENTOS = "Consolidação de eventos"
    FILTRAGEM_LIMIAR = "Filtragem por limiar"
    FLAG_REPETICAO = "Flag de repetição"
    
    # Dados Estruturados
    REFORCO_CONSTRAINTS = "Reforço de constraints"
    NORMALIZACAO_ESQUEMA = "Normalização de esquema"
    CORRECAO_CHAVES = "Correção de chaves"
    INTEGRIDADE_REFERENCIAL = "Implementação de integridade referencial"
    
    # Dados Semi-estruturados
    REESTRUTURACAO_DADOS = "Reestruturação de dados"
    NORMALIZACAO_FORMATO = "Normalização de formato"
    DEDUPLICACAO_SUBELEMENTOS = "Deduplicação de sub-elementos"
    CONSOLIDACAO_HIERARQUICA = "Consolidação hierárquica"
    
    # Duplicatas Transacionais
    PROCESSAMENTO_IDEMPOTENTE = "Processamento idempotente"
    REGISTRO_TRANSACOES = "Registro de transações processadas"
    CONFIRMACAO_TRANSACAO = "Confirmação de transação"
    ROLLBACK_DUPLICATAS = "Rollback de duplicatas"
    
    # Migração de Dados
    MIGRACAO_FASEADA = "Estratégia de migração faseada"
    VERIFICACAO_PREPOS = "Verificação pré/pós migração"
    RECONCILIACAO_DADOS = "Reconciliação de dados"
    LIMPEZA_POS_MIGRACAO = "Limpeza pós-migração"
    
    # Usuários/Entidades
    MERGE_PERFIS = "Merge de perfis"
    NOTIFICACAO_USUARIO = "Notificação ao usuário"
    CONSOLIDACAO_PROGRESSIVA = "Consolidação progressiva"
    SISTEMA_MDM = "Sistema MDM"
    
    # Big Data
    DEDUPLICACAO_DISTRIBUIDA = "MapReduce para deduplicação"
    PROCESSAMENTO_DISTRIBUIDO = "Processamento distribuído"
    SHARDING_CHAVES = "Sharding por chaves"
    DEDUPLICACAO_STREAM = "Deduplicação em stream"
    
    # Dados Textuais
    AGRUPAMENTO_SIMILARIDADE = "Agrupamento por similaridade"
    REPRESENTACAO_CANONICA = "Representação canônica"
    CITACAO_FONTES = "Citação de fontes"
    DETECCAO_VARIACOES = "Detecção de variações"
    
    # Machine Learning
    PARTICIONAMENTO_ESTRATIFICADO = "Particionamento estratificado"
    REMOCAO_BALANCEADA = "Remoção balanceada"
    VERIFICACAO_VAZAMENTO = "Verificação de vazamento"
    AMOSTRAGEM_REPRESENTATIVA = "Amostragem representativa"

class DuplicatesScenario(Enum):
    """Cenários de duplicatas com soluções, descrição e ferramentas correspondentes"""
    
    # Duplicatas Exatas
    EXATAS = (
        [
            DuplicatesSolution.REMOCAO_DUPLICATAS,
            DuplicatesSolution.MANUTENCAO_PRIMEIRO_ULTIMO,
            DuplicatesSolution.AGREGACAO_POR_ID,
            DuplicatesSolution.FLAGS_DUPLICACAO
        ],
        "Registros idênticos em todas as colunas. Desafios: Volume de dados, performance em grandes datasets, coordenação entre sistemas.",
    )
    
    # Duplicatas Parciais
    PARCIAIS = (
        [
            DuplicatesSolution.PRIORIZACAO_REGISTROS_COMPLETOS,
            DuplicatesSolution.MESCLAGEM_INFORMACOES,
            DuplicatesSolution.RESOLUCAO_BASEADA_REGRA,
            DuplicatesSolution.EXCLUSAO_AMBIGUIDADE
        ],
        "Registros similares com diferenças parciais. Desafios: Definição de regras de mesclagem, informações conflitantes.",
    )
    
    # Duplicatas Fuzzy (Aproximadas)
    FUZZY = (
        [
            DuplicatesSolution.CLUSTERING_SIMILARIDADE,
            DuplicatesSolution.CORRECAO_MANUAL_ASSISTIDA,
            DuplicatesSolution.CONSOLIDACAO_LIMIAR,
            DuplicatesSolution.DEDUPLICACAO_ITERATIVA
        ],
        "Registros similares com variações textuais. Desafios: Definição de limiares, escalabilidade, complexidade computacional.",
    )
    
    # Duplicatas Multi-Fonte
    MULTI_FONTE = (
        [
            DuplicatesSolution.SISTEMA_ENTIDADE_MESTRA,
            DuplicatesSolution.HIERARQUIA_FONTES,
            DuplicatesSolution.TIMESTAMPS_ATUALIZACAO,
            DuplicatesSolution.CONSOLIDACAO_REGRA_NEGOCIO
        ],
        "Dados duplicados de múltiplas fontes. Desafios: Diferentes formatos, esquemas distintos, conflitos de valores.",
    )
    
    # Duplicatas com Valores Diferentes
    VALORES_DIFERENTES = (
        [
            DuplicatesSolution.RESOLUCAO_REGRA_CONFLITO,
            DuplicatesSolution.REGISTRO_MAIS_RECENTE,
            DuplicatesSolution.CONSOLIDACAO_COMPLETUDE,
            DuplicatesSolution.CRIACAO_VERSAO
        ],
        "Registros com mesma chave mas valores conflitantes. Desafios: Identificação do valor correto, registros parcialmente atualizados.",
    )
    
    # Duplicatas Hierárquicas
    HIERARQUICAS = (
        [
            DuplicatesSolution.NORMALIZACAO_HIERARQUIA,
            DuplicatesSolution.CONSOLIDACAO_NIVEIS,
            DuplicatesSolution.RESOLUCAO_TOP_DOWN,
            DuplicatesSolution.RESTRUTURACAO_ARVORES
        ],
        "Duplicatas em estruturas hierárquicas. Desafios: Relações complexas, propagação de mudanças, ciclos.",
    )
    
    # Duplicatas Temporais
    TEMPORAIS = (
        [
            DuplicatesSolution.AGREGACAO_TEMPORAL,
            DuplicatesSolution.REMOCAO_JANELA,
            DuplicatesSolution.AMOSTRAGEM_CONTROLADA,
            DuplicatesSolution.CONSOLIDACAO_INTERVALO
        ],
        "Registros duplicados em séries temporais. Desafios: Granularidade temporal, fusos horários.",
    )
    
    # Duplicatas de Eventos
    EVENTOS = (
        [
            DuplicatesSolution.DEDUPLICACAO_JANELA,
            DuplicatesSolution.CONSOLIDACAO_EVENTOS,
            DuplicatesSolution.FILTRAGEM_LIMIAR,
            DuplicatesSolution.FLAG_REPETICAO
        ],
        "Eventos duplicados em fluxos de dados. Desafios: Definição de evento único, causalidade vs correlação.",
    )
    
    # Duplicatas em Dados Estruturados
    ESTRUTURADOS = (
        [
            DuplicatesSolution.REFORCO_CONSTRAINTS,
            DuplicatesSolution.NORMALIZACAO_ESQUEMA,
            DuplicatesSolution.CORRECAO_CHAVES,
            DuplicatesSolution.INTEGRIDADE_REFERENCIAL
        ],
        "Duplicatas em bancos de dados relacionais. Desafios: Schemas flexíveis, sistemas distribuídos.",
    )
    
    # Duplicatas em Dados Semi-estruturados
    SEMI_ESTRUTURADOS = (
        [
            DuplicatesSolution.REESTRUTURACAO_DADOS,
            DuplicatesSolution.NORMALIZACAO_FORMATO,
            DuplicatesSolution.DEDUPLICACAO_SUBELEMENTOS,
            DuplicatesSolution.CONSOLIDACAO_HIERARQUICA
        ],
        "Duplicatas em JSON/XML. Desafios: Estruturas aninhadas, evolução de schema.",
    )
    
    # Duplicatas Transacionais
    TRANSACIONAIS = (
        [
            DuplicatesSolution.PROCESSAMENTO_IDEMPOTENTE,
            DuplicatesSolution.REGISTRO_TRANSACOES,
            DuplicatesSolution.CONFIRMACAO_TRANSACAO,
            DuplicatesSolution.ROLLBACK_DUPLICATAS
        ],
        "Transações duplicadas em sistemas distribuídos. Desafios: Falhas de rede, consistência eventual.",
    )
    
    # Duplicatas em Migração de Dados
    MIGRACAO = (
        [
            DuplicatesSolution.MIGRACAO_FASEADA,
            DuplicatesSolution.VERIFICACAO_PREPOS,
            DuplicatesSolution.RECONCILIACAO_DADOS,
            DuplicatesSolution.LIMPEZA_POS_MIGRACAO
        ],
        "Duplicatas geradas durante migração. Desafios: Diferenças de schema, janelas de manutenção.",
    )
    
    # Duplicatas de Usuários/Entidades
    USUARIOS = (
        [
            DuplicatesSolution.MERGE_PERFIS,
            DuplicatesSolution.NOTIFICACAO_USUARIO,
            DuplicatesSolution.CONSOLIDACAO_PROGRESSIVA,
            DuplicatesSolution.SISTEMA_MDM
        ],
        "Perfis duplicados de usuários/entidades. Desafios: Privacidade, fragmentação de informação.",
    )
    
    # Duplicatas em Big Data
    BIG_DATA = (
        [
            DuplicatesSolution.DEDUPLICACAO_DISTRIBUIDA,
            DuplicatesSolution.PROCESSAMENTO_DISTRIBUIDO,
            DuplicatesSolution.SHARDING_CHAVES,
            DuplicatesSolution.DEDUPLICACAO_STREAM
        ],
        "Duplicatas em grandes volumes distribuídos. Desafios: Escalabilidade, precisão vs tempo.",
    )
    
    # Duplicatas em Dados Textuais
    TEXTUAIS = (
        [
            DuplicatesSolution.AGRUPAMENTO_SIMILARIDADE,
            DuplicatesSolution.REPRESENTACAO_CANONICA,
            DuplicatesSolution.CITACAO_FONTES,
            DuplicatesSolution.DETECCAO_VARIACOES
        ],
        "Documentos/textos similares. Desafios: Similaridade semântica, plágio vs citação.",
    )
    
    # Duplicatas em Machine Learning
    ML = (
        [
            DuplicatesSolution.PARTICIONAMENTO_ESTRATIFICADO,
            DuplicatesSolution.REMOCAO_BALANCEADA,
            DuplicatesSolution.VERIFICACAO_VAZAMENTO,
            DuplicatesSolution.AMOSTRAGEM_REPRESENTATIVA
        ],
        "Vazamento de dados entre conjuntos de treino/teste. Desafios: Overfitting, generalização.",
    )

    def __init__(self, 
                solucoes: List[DuplicatesSolution],
                descricao: str):
        self.solucoes = solucoes
        self.descricao = descricao

class DuplicatesAnalyzer(AnalysisStep):
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> dict:
        pass