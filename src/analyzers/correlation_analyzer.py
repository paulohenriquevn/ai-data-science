from enum import Enum
from abc import abstractmethod
from src.analyzers.analysis_step import AnalysisStep
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from src.utils import detect_and_replace_placeholders

class CorrelationTechnique(Enum):
    """Técnicas de análise de correlação entre variáveis."""
    PEARSON = "Correlação de Pearson"
    SPEARMAN = "Correlação de Spearman"
    KENDALL = "Correlação de Kendall"
    POINT_BISERIAL = "Correlação Point-Biserial"
    PHI_COEFFICIENT = "Coeficiente Phi"
    CRAMERS_V = "V de Cramer"
    MUTUAL_INFORMATION = "Informação Mútua"
    MAXIMAL_INFORMATION_COEFFICIENT = "Coeficiente de Informação Máxima"
    DISTANCE_CORRELATION = "Correlação de Distância"
    CANONICAL_CORRELATION = "Correlação Canônica"
    
    # Técnicas adicionais
    PARTIAL_CORRELATION = "Correlação Parcial"
    SEMI_PARTIAL_CORRELATION = "Correlação Semi-Parcial"
    POLYCHORIC_CORRELATION = "Correlação Policórica"
    TETRACHORIC_CORRELATION = "Correlação Tetracórica"
    BIWEIGHT_MIDCORRELATION = "Correlação Biweight Midweight"
    RANK_BISERIAL_CORRELATION = "Correlação Rank-Biserial"
    
    # Técnicas para séries temporais
    CROSS_CORRELATION = "Correlação Cruzada"
    COHERENCE = "Coerência"
    WAVELET_COHERENCE = "Coerência Wavelet"
    DYNAMIC_TIME_WARPING = "Dynamic Time Warping"
    GRANGER_CAUSALITY = "Causalidade de Granger"
    
    # Técnicas multivariadas
    PRINCIPAL_COMPONENT_ANALYSIS = "Análise de Componentes Principais"
    FACTOR_ANALYSIS = "Análise Fatorial"
    STRUCTURAL_EQUATION_MODELING = "Modelagem de Equações Estruturais"
    
    # Técnicas não-lineares
    NONLINEAR_CORRELATION = "Correlação Não-Linear"
    MAXIMAL_CORRELATION = "Correlação Máxima"
    HOEFFDING_D = "D de Hoeffding"
    DISTANCE_COVARIANCE = "Covariância de Distância"
    
    # Técnicas de aprendizado de máquina
    RANDOM_FOREST_IMPORTANCE = "Importância de Random Forest"
    GRADIENT_BOOSTING_IMPORTANCE = "Importância de Gradient Boosting"
    PERMUTATION_IMPORTANCE = "Importância por Permutação"
    
    # Modelos econométricos
    PANEL_DATA_MODELS = "Modelos de Dados em Painel"
    TIME_SERIES_MODELS = "Modelos de Séries Temporais"
    VECTOR_AUTOREGRESSION = "Vetores Autorregressivos (VAR)"
    REGIME_SWITCHING_MODELS = "Modelos de Mudança de Regime"
    REGIME_SWITCH_MODELS = "Modelos de Mudança de Regime (alternativo)"
    
    # Técnicas bayesianas
    BAYESIAN_CORRELATION = "Correlação Bayesiana"
    BAYESIAN_NETWORK = "Rede Bayesiana"
    
    # Técnicas de visualização
    SCATTER_PLOT = "Gráfico de Dispersão"
    HEAT_MAP = "Mapa de Calor"
    NETWORK_GRAPH = "Grafo de Rede"
    CHORD_DIAGRAM = "Diagrama de Acordes"
    
    # Técnicas adicionais mencionadas em CorrelationScenario
    FEATURE_SELECTION = "Seleção de Features"
    REMOVE_VARIABLE = "Remover Variável"
    CREATE_COMPOSITE_INDICATORS = "Criar Indicadores Compostos"
    REGULARIZATION = "Regularização"
    PCA = "PCA"
    STEPWISE_SELECTION = "Seleção Stepwise"
    VARIABLE_REDEFINITION = "Redefinição de Variável"
    DIFFERENTIATION = "Diferenciação"
    ARIMA_MODELING = "Modelagem ARIMA"
    GLS = "GLS"
    LINEAR_TRANSFORMATION = "Transformação Linear"
    NON_PARAMETRIC_METHODS = "Métodos Não-Paramétricos"
    POLYNOMIAL_EXPANSION = "Expansão Polinomial"
    TWO_STAGE_ANALYSIS = "Análise em Duas Etapas"
    STRUCTURAL_MODELING = "Modelagem Estrutural"
    LATENT_INDICATORS = "Indicadores Latentes"
    INCLUDE_CONTROLS = "Incluir Controles"
    MATCHING_METHODS = "Métodos de Matching"
    CAUSAL_ANALYSIS = "Análise Causal"
    SPECIFIC_ENCODING = "Codificação Específica"
    CATEGORY_COMBINATION = "Combinação de Categorias"
    CORRESPONDENCE_ANALYSIS = "Análise de Correspondência"
    TARGET_BASED_SELECTION = "Seleção Baseada no Target"
    WRAPER_METHODS = "Métodos Wrapper"
    TREE_IMPORTANCE = "Importância em Árvores"
    VARIABLE_PARAM_MODELS = "Modelos de Parâmetros Variáveis"
    TEMPORAL_SEGMENTATION = "Segmentação Temporal"
    ELASTIC_NET = "Elastic Net"
    ENSEMBLE_METHODS = "Métodos Ensemble"
    FEATURE_CLUSTERING = "Agrupamento de Features"
    VARIABLE_TRANSFORMATION = "Transformação de Variáveis"
    ROBUST_SE = "Erros Padrão Robustos"
    WLS = "WLS"
    INTERACTION_TERMS = "Termos de Interação"
    GAM = "GAM"
    FIXED_EFFECTS = "Efeitos Fixos"
    WITHIN_TRANSFORMATION = "Transformação Within"
    SPATIAL_AUTOREGRESSIVE = "Autorregressão Espacial"
    GEO_WEIGHTED_REGRESSION = "Regressão Ponderada Geograficamente"
    KRIGING = "Kriging"

class CorrelationScenario(Enum):
    """Associa cada tipo/cenário às técnicas de tratamento e descrições correspondentes."""
    
    # Correlação Linear
    LINEAR_CORRELATION = (
        [
            CorrelationTechnique.FEATURE_SELECTION,
            CorrelationTechnique.REMOVE_VARIABLE,
            CorrelationTechnique.CREATE_COMPOSITE_INDICATORS
        ],
        "|r| > 0.7 (forte). Impacto: Inflação de variância, instabilidade nos coeficientes"
    )
    
    # Multicolinearidade
    MULTICOLLINEARITY = (
        [
            CorrelationTechnique.REGULARIZATION,
            CorrelationTechnique.PCA,
            CorrelationTechnique.STEPWISE_SELECTION
        ],
        "VIF > 10. Impacto: Instabilidade nos coeficientes, erros padrão elevados"
    )
    
    # Colinearidade Perfeita
    PERFECT_COLLINEARITY = (
        [
            CorrelationTechnique.REMOVE_VARIABLE,
            CorrelationTechnique.VARIABLE_REDEFINITION
        ],
        "Correlação exata (r = ±1). Impacto: Impossibilidade de estimar o modelo"
    )
    
    # Correlação Serial
    SERIAL_CORRELATION = (
        [
            CorrelationTechnique.DIFFERENTIATION,
            CorrelationTechnique.ARIMA_MODELING,
            CorrelationTechnique.GLS
        ],
        "DW < 1 ou > 3. Típica em séries temporais"
    )
    
    # Correlação Não-Linear
    NONLINEAR_CORRELATION = (
        [
            CorrelationTechnique.LINEAR_TRANSFORMATION,
            CorrelationTechnique.NON_PARAMETRIC_METHODS,
            CorrelationTechnique.POLYNOMIAL_EXPANSION
        ],
        "Detectada com Spearman/η. Requer modelagem não-linear"
    )
    
    # Correlação Moderada
    MODERATE_CORRELATION = (
        [
            CorrelationTechnique.FEATURE_SELECTION,
            CorrelationTechnique.VARIABLE_TRANSFORMATION,
            CorrelationTechnique.INTERACTION_TERMS
        ],
        "0.3 ≤ |r| < 0.7. Associação razoável, útil para modelagem"
    )
    
    # Sem Correlação
    NO_CORRELATION = (
        [
            CorrelationTechnique.REMOVE_VARIABLE,
            CorrelationTechnique.NONLINEAR_CORRELATION,
            CorrelationTechnique.POLYNOMIAL_EXPANSION
        ],
        "|r| < 0.15. Ausência de relação linear, verificar relações não-lineares"
    )
    
    # Colinearidade Estrutural
    STRUCTURAL_COLLINEARITY = (
        [CorrelationTechnique.TWO_STAGE_ANALYSIS, CorrelationTechnique.STRUCTURAL_MODELING, CorrelationTechnique.LATENT_INDICATORS],
        "VIF por grupo elevado. Impacto: Confusão entre efeitos"
    )
    
    # Correlação Espúria
    SPURIOUS_CORRELATION = (
        [CorrelationTechnique.INCLUDE_CONTROLS, CorrelationTechnique.MATCHING_METHODS, CorrelationTechnique.CAUSAL_ANALYSIS],
        "Mudança na correlação após controle. Impacto: Conclusões errôneas"
    )
    
    # Correlação Categórica
    CATEGORICAL_CORRELATION = (
        [CorrelationTechnique.SPECIFIC_ENCODING, CorrelationTechnique.CATEGORY_COMBINATION, CorrelationTechnique.CORRESPONDENCE_ANALYSIS],
        "V de Cramer > 0.25. Impacto: Problemas com dummies"
    )
    
    # Correlação com Target
    TARGET_CORRELATION = (
        [CorrelationTechnique.TARGET_BASED_SELECTION, CorrelationTechnique.WRAPER_METHODS, CorrelationTechnique.TREE_IMPORTANCE],
        "IV > 0.3 (forte). Fundamental para feature selection"
    )
    
    # Não-Estacionariedade
    NON_STATIONARY = (
        [CorrelationTechnique.VARIABLE_PARAM_MODELS, CorrelationTechnique.TEMPORAL_SEGMENTATION, CorrelationTechnique.REGIME_SWITCH_MODELS],
        "Mudanças temporais. Comum em finanças/macroeconomia"
    )
    
    # Alta Dimensão
    HIGH_DIMENSIONALITY = (
        [CorrelationTechnique.ELASTIC_NET, CorrelationTechnique.ENSEMBLE_METHODS, CorrelationTechnique.FEATURE_CLUSTERING],
        "VIF em alta dimensão. Impacto: Curse of dimensionality"
    )
    
    # Heteroscedasticidade
    HETEROSCEDASTICITY = (
        [CorrelationTechnique.VARIABLE_TRANSFORMATION, CorrelationTechnique.ROBUST_SE, CorrelationTechnique.WLS],
        "Padrão em cone nos resíduos. Impacto: Inferência distorcida"
    )
    
    # Interações
    INTERACTION_EFFECTS = (
        [CorrelationTechnique.INTERACTION_TERMS, CorrelationTechnique.GAM],
        "Melhoria com interações. Impacto: Risco de overfitting"
    )
    
    # Dados Longitudinais
    LONGITUDINAL_DATA = (
        [CorrelationTechnique.FIXED_EFFECTS, CorrelationTechnique.WITHIN_TRANSFORMATION],
        "Componentes intra-grupo. Requer modelos painel"
    )
    
    # Correlação Espacial
    SPATIAL_CORRELATION = (
        [CorrelationTechnique.SPATIAL_AUTOREGRESSIVE, CorrelationTechnique.GEO_WEIGHTED_REGRESSION, CorrelationTechnique.KRIGING],
        "Índice de Moran > 0. Típica em dados geográficos"
    )
    # Demais cenários omitidos por brevidade. Padrão continua similar...

    def __init__(self, solutions: list[CorrelationTechnique], description: str):
        self.solutions = solutions
        self.description = description



class CorrelationAnalyzer(AnalysisStep):
    """
    Etapa de análise de correlação entre variáveis numéricas e com a variável alvo
    """

    def __init__(self, target: str = 'y', threshold: float = 0.6):
        self.target = target
        self.threshold = threshold

    def analyze(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        df = detect_and_replace_placeholders(data)
        
        numeric_df = df.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()

        results = []

        # Correlação com a variável alvo
        corr_with_target = corr_matrix[self.target].drop(self.target).sort_values(ascending=False)
        for col, corr in corr_with_target.items():
            sentido = 'positivo' if corr >= 0 else 'negativo'
            abs_corr = abs(corr)

            if corr >= 0.6:
                problem = 'CORRELACAO_FORTE_POSITIVA'
                scenario = CorrelationScenario.TARGET_CORRELATION
                nivel = 'alta'
            elif corr <= -0.6:
                problem = 'CORRELACAO_FORTE_NEGATIVA'
                scenario = CorrelationScenario.TARGET_CORRELATION
                nivel = 'alta'
            elif abs_corr >= 0.3:
                problem = 'CORRELACAO_MODERADA'
                scenario = CorrelationScenario.MODERATE_CORRELATION
                nivel = 'moderada'
            elif abs_corr < 0.15:
                problem = 'SEM_CORRELACAO'
                scenario = CorrelationScenario.NO_CORRELATION
                nivel = 'baixa'
            else:
                continue

            results.append({
                'column': col,
                'problem': problem,
                'problem_description': scenario.description,
                'description': scenario.description,
                'solution': scenario.solutions[0].name if scenario.solutions else None,
                'actions': [t.name for t in scenario.solutions] if scenario.solutions else [],
                'statistics': {
                    'correlation_with_target': round(corr, 3),
                    'sentido': sentido,
                    'nivel_relevancia': nivel
                }
            })

        # Análise entre pares de variáveis
        seen = set()
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if j <= i or col1 == self.target or col2 == self.target:
                    continue
                pair = tuple(sorted([col1, col2]))
                if pair in seen:
                    continue
                seen.add(pair)
                corr_value = corr_matrix.loc[col1, col2]
                sentido = 'positivo' if corr_value >= 0 else 'negativo'
                abs_corr = abs(corr_value)

                if corr_value >= self.threshold:
                    problem = 'MULTICOLINEARIDADE_POSITIVA'
                    scenario = CorrelationScenario.MULTICOLLINEARITY
                    nivel = 'alta'
                elif corr_value <= -self.threshold:
                    problem = 'MULTICOLINEARIDADE_NEGATIVA'
                    scenario = CorrelationScenario.MULTICOLLINEARITY
                    nivel = 'alta'
                elif 0.3 <= abs_corr < self.threshold:
                    problem = 'CORRELACAO_MODERADA'
                    scenario = CorrelationScenario.MODERATE_CORRELATION
                    nivel = 'moderada'
                elif abs_corr < 0.05:
                    problem = 'SEM_CORRELACAO'
                    scenario = CorrelationScenario.NO_CORRELATION
                    nivel = 'baixa'
                else:
                    continue

                results.append({
                    'column': f'{col1} ~ {col2}',
                    'problem': problem,
                    'problem_description': scenario.description,
                    'description': scenario.description,
                    'solution': scenario.solutions[0].name if scenario.solutions else None,
                    'actions': [t.name for t in scenario.solutions] if scenario.solutions else [],
                    'statistics': {
                        'correlation': round(corr_value, 3),
                        'sentido': sentido,
                        'nivel_relevancia': nivel
                    }
                })

        return results

