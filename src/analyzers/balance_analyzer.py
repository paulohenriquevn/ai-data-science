from enum import Enum
import pandas as pd
from src.analyzers.analysis_step import AnalysisStep

class BalanceTechnique(Enum):
    """Técnicas de balanceamento mapeadas do documento."""
    RANDOM_UNDERSAMPLING = "Random Undersampling"
    NEARMISS = "NearMiss"
    TOMEK_LINKS = "Tomek Links"
    CONDENSED_NEAREST_NEIGHBORS = "Condensed Nearest Neighbors (CNN)"
    RANDOM_OVERSAMPLING = "Random Oversampling"
    SMOTE = "SMOTE"
    BORDERLINE_SMOTE = "Borderline-SMOTE"
    ADASYN = "ADASYN"
    SMOTE_TOMEK_LINKS = "SMOTE + Tomek Links"
    SMOTETOMEK = "SMOTETomek"
    SMOTE_ENN = "SMOTE + ENN"
    CLASS_WEIGHTS = "Class Weights"
    COST_SENSITIVE_LEARNING = "Cost-Sensitive Learning"
    ENSEMBLE_BALANCEAMENTO = "Ensemble com Balanceamento"
    BALANCED_BAGGING = "Bagging-based (BalancedBaggingClassifier)"
    EASY_ENSEMBLE = "EasyEnsemble"
    RUSBOOST = "RUSBoost"
    THRESHOLD_MOVING = "Threshold Moving"
    ONE_CLASS_LEARNING = "One-Class Learning"

class BalanceScenario(Enum):
    """Associação direta entre cenários e técnicas conforme o documento."""
    
    # Undersampling
    DATASET_GRANDE_MAIORIA = (
        BalanceTechnique.RANDOM_UNDERSAMPLING, 
        "Datasets muito grandes com muitos exemplos da classe majoritária"
    )
    FRONTEIRA_SEMANTICA_ESPACIAL = (
        BalanceTechnique.NEARMISS, 
        "Fronteira de decisão importante ou semântica espacial relevante"
    )
    LIMPEZA_PRE_PROCESSAMENTO = (
        BalanceTechnique.TOMEK_LINKS, 
        "Limpeza de dados ou pré-processamento combinado com outras técnicas"
    )
    REDUCAO_SUBSTANCIAL_DADOS = (
        BalanceTechnique.CONDENSED_NEAREST_NEIGHBORS, 
        "Redução substancial de dados com fronteira bem definida"
    )
    
    # Oversampling
    POUCOS_EXEMPLOS_MINORITARIOS = (
        BalanceTechnique.RANDOM_OVERSAMPLING, 
        "Conjuntos pequenos ou perda significativa com undersampling"
    )
    EXPANSAO_ESPACO_CARACTERISTICAS = (
        BalanceTechnique.SMOTE, 
        "Poucos exemplos minoritários e dados numéricos"
    )
    FRONTEIRA_DECISAO_CRITICA = (
        BalanceTechnique.BORDERLINE_SMOTE, 
        "Fronteira de decisão crítica para discriminação entre classes"
    )
    REGIOES_DIFICEIS_NAO_UNIFORMES = (
        BalanceTechnique.ADASYN, 
        "Distribuição não uniforme com muitas regiões difíceis"
    )
    
    # Híbridos
    SOBREPOSICAO_CLASSES = (
        BalanceTechnique.SMOTE_TOMEK_LINKS, 
        "Datasets com sobreposição entre classes"
    )
    WORKFLOW_PADRONIZADO = (
        BalanceTechnique.SMOTETOMEK, 
        "Workflow padronizado usando bibliotecas como imbalanced-learn"
    )
    DADOS_RUIDOSOS_AGRESIVO = (
        BalanceTechnique.SMOTE_ENN, 
        "Datasets com muito ruído e limpeza prioritária"
    )
    
    # Algoritmo
    PRESERVA_DADOS_ORIGINAIS = (
        BalanceTechnique.CLASS_WEIGHTS, 
        "Algoritmos que suportam pesos (ex: SVM, Random Forest)"
    )
    CUSTOS_ERRO_ESPECIFICOS = (
        BalanceTechnique.COST_SENSITIVE_LEARNING, 
        "Custos de erro conhecidos (ex: fraude, diagnóstico)"
    )
    
    # Ensemble
    MAXIMIZAR_PERFORMANCE = (
        BalanceTechnique.ENSEMBLE_BALANCEAMENTO, 
        "Conjuntos grandes com recursos computacionais disponíveis"
    )
    VARIANCIA_ALTA_UNDERSAMPLING = (
        BalanceTechnique.BALANCED_BAGGING, 
        "Variância alta após undersampling em datasets médios/grandes"
    )
    PROBLEMAS_COMPLEXOS_RECURSOS = (
        BalanceTechnique.EASY_ENSEMBLE, 
        "Problemas complexos com recursos computacionais"
    )
    GRANDES_DATASETS_EFICIENCIA = (
        BalanceTechnique.RUSBOOST, 
        "Grandes datasets onde random undersampling funciona bem"
    )
    
    # Pós-processamento
    AJUSTE_LIMIAR_PROBABILIDADE = (
        BalanceTechnique.THRESHOLD_MOVING, 
        "Ajuste de limiar pós-treinamento com probabilidades calibradas"
    )
    
    # Casos extremos
    CLASSES_EXTREMAMENTE_RARAS = (
        BalanceTechnique.ONE_CLASS_LEARNING, 
        "Classes extremamente desbalanceadas (< 1%) ou detecção de anomalias"
    )

    def __init__(self, technique: BalanceTechnique, description: str):
        self.technique = technique
        self.description = description
    
class BalanceSolution(Enum):
    pass

class BalanceAnalyzer(AnalysisStep):
    def analyze(self, data: pd.DataFrame) -> dict:
        # Implementar lógica de análise de balanceamento
        return {
            'summary': 'Dataset desbalanceado (proporção 1:10)',
            'suggested_techniques': ['SMOTE', 'Class Weights']
        }