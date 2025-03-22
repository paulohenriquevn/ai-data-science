from enum import Enum
from abc import abstractmethod
from src.analyzers.analysis_step import AnalysisStep
from typing import List, Dict, Any
import pandas as pd
import numpy as np

class CorrelationTechnique(Enum):
    """Enumera todas as técnicas de tratamento de correlação do documento."""
    FEATURE_SELECTION = "Seleção de features"
    REMOVE_VARIABLE = "Remoção de uma das variáveis"
    CREATE_COMPOSITE_INDICATORS = "Criação de indicadores compostos"
    LINEAR_TRANSFORMATION = "Transformações para linearizar relações"
    NON_PARAMETRIC_METHODS = "Métodos não paramétricos"
    POLYNOMIAL_EXPANSION = "Expansão polinomial"
    DIFFERENTIATION = "Diferenciação"
    ARIMA_MODELING = "Modelagem ARIMA"
    LAGS_AS_PREDICTORS = "Inclusão de lags como preditores"
    GLS = "GLS (Mínimos Quadrados Generalizados)"
    REGULARIZATION = "Regularização (Ridge, Lasso)"
    PCA = "Análise de Componentes Principais (PCA)"
    STEPWISE_SELECTION = "Seleção stepwise"
    VARIABLE_REDEFINITION = "Redefinição de variáveis"
    TWO_STAGE_ANALYSIS = "Análise em dois estágios"
    HIERARCHICAL_MODELING = "Modelagem hierárquica"
    LATENT_INDICATORS = "Indicadores latentes"
    INCLUDE_CONTROLS = "Inclusão de controles"
    MATCHING_METHODS = "Métodos de pareamento"
    INSTRUMENTAL_VARIABLES = "Variáveis instrumentais"
    MEDIATION_ANALYSIS = "Análise de mediação"
    SPECIFIC_ENCODING = "Encoding específico"
    CATEGORY_COMBINATION = "Combinação de categorias"
    CORRESPONDENCE_ANALYSIS = "Análise de correspondência"
    TARGET_BASED_SELECTION = "Seleção baseada em correlação"
    WRAPPER_METHODS = "Métodos wrapper"
    TREE_IMPORTANCE = "Importância em modelos de árvore"
    VARIABLE_PARAM_MODELS = "Modelagem com parâmetros variáveis"
    TIME_SEGMENTATION = "Segmentação temporal"
    REGIME_SWITCH_MODELS = "Modelos de mudança de regime"
    ELASTIC_NET = "Elastic Net"
    ENSEMBLE_METHODS = "Métodos ensemble"
    RECURSIVE_SELECTION = "Seleção recursiva"
    FEATURE_CLUSTERING = "Agrupamento de features"
    VARIABLE_TRANSFORMATION = "Transformação de variáveis"
    ROBUST_SE = "Erros padrão robustos"
    WLS = "Modelos ponderados (WLS)"
    INTERACTION_TERMS = "Inclusão de termos de interação"
    GAM = "GAM (Modelos Aditivos Generalizados)"
    FIXED_EFFECTS = "Modelos de efeitos fixos/aleatórios"
    WITHIN_TRANSFORMATION = "Transformação within"
    SPATIAL_MODELS = "Modelos espaciais (SAR, SEM)"
    GEO_WEIGHTED_REGRESSION = "Geographically Weighted Regression"
    KRIGING = "Kriging"
    SPATIAL_AUTOREGRESSIVE = "Modelo espacial auto regressivo"
    CAUSAL_ANALYSIS = "Análise causal"
    STRUCTURAL_MODELING = "Modelagem estrutural"
    TEMPORAL_SEGMENTATION = "Segmentação temporal"
    REGIME_SWITCH_MODELS = "Modelos de mudança de regime"
    ELASTIC_NET = "Elastic Net"
    ENSEMBLE_METHODS = "Métodos ensemble"
    FEATURE_CLUSTERING = "Agrupamento de features"
    VARIABLE_TRANSFORMATION = "Transformação de variáveis"
    WRAPER_METHODS = "Métodos wrapper"

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
        df = data.replace(-999, np.nan)
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

