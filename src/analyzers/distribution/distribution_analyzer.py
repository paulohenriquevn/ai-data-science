from enum import Enum
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
from scipy import stats
from src.analyzers.base.analysis_base import AnalysisStep
from src.utils import detect_and_replace_placeholders

class DistributionType(Enum):
    """Tipos de distribuições estatísticas"""
    NORMAL = "Distribuição Normal"
    ASSIMETRICA_POSITIVA = "Distribuição Assimétrica Positiva"
    ASSIMETRICA_NEGATIVA = "Distribuição Assimétrica Negativa"
    UNIFORME = "Distribuição Uniforme"
    LOGNORMAL = "Distribuição Logarítmica Normal"
    EXPONENCIAL = "Distribuição Exponencial"
    CAUDA_PESADA = "Distribuição de Cauda Pesada"
    BIMODAL = "Distribuição Bimodal"
    MULTIMODAL = "Distribuição Multimodal"
    POISSON = "Distribuição de Poisson"
    BINOMIAL = "Distribuição Binomial"
    DESCONHECIDA = "Distribuição Desconhecida"


class DistributionSolution(Enum):
    """Soluções para problemas de distribuição dos dados"""
    TRANSFORMACAO_LOG = "Transformação logarítmica"
    TRANSFORMACAO_RAIZ_QUADRADA = "Transformação raiz quadrada"
    TRANSFORMACAO_BOX_COX = "Transformação Box-Cox"
    TRANSFORMACAO_YEOJONHSON = "Transformação Yeo-Johnson"
    TRANSFORMACAO_REFLECAO_LOG = "Reflexão e transformação logarítmica"
    TRANSFORMACAO_QUADRATICA = "Transformação quadrática (x²)"
    TRANSFORMACAO_INVERSA = "Transformação inversa (1/x)"
    CENTRALIZACAO_PADRONIZACAO = "Centralização e padronização (Z-score)"
    WINSORIZATION = "Winsorização (limitar extremos)"
    CLIPPING = "Clipping de valores extremos"
    ALGORITMO_ROBUSTO = "Usar algoritmo robusto a distribuições assimétricas"
    QUANTIZATION = "Quantização (converter em categorias)"
    NENHUMA = "Nenhuma transformação necessária"
    REMOVER_OUTLIERS = "Remover outliers"


# Alterando para uma classe diferente para evitar conflito de nomes
class DistributionPattern(Enum):
    """Cenários de distribuição e suas respectivas soluções"""
    NORMAL_PADRAO = (
        "Distribuição aproximadamente normal",
        "Os dados seguem uma distribuição aproximadamente normal, sem assimetria significativa.",
        [] # Nenhuma ação necessária
    )
    
    NORMAL_COM_OUTLIERS = (
        "Distribuição normal com outliers",
        "Os dados seguem uma distribuição aproximadamente normal, mas com presença de outliers.",
        [DistributionSolution.REMOVER_OUTLIERS, DistributionSolution.WINSORIZATION]
    )
    
    ASSIMETRIA_POSITIVA_LEVE = (
        "Assimetria positiva leve",
        "Os dados possuem uma assimetria positiva (cauda à direita) leve.",
        [DistributionSolution.TRANSFORMACAO_RAIZ_QUADRADA]
    )
    
    ASSIMETRIA_POSITIVA_MODERADA = (
        "Assimetria positiva moderada",
        "Os dados possuem uma assimetria positiva (cauda à direita) moderada.",
        [DistributionSolution.TRANSFORMACAO_LOG]
    )
    
    ASSIMETRIA_POSITIVA_FORTE = (
        "Assimetria positiva forte",
        "Os dados possuem uma assimetria positiva (cauda à direita) forte.",
        [DistributionSolution.TRANSFORMACAO_LOG, DistributionSolution.TRANSFORMACAO_BOX_COX]
    )
    
    ASSIMETRIA_NEGATIVA_LEVE = (
        "Assimetria negativa leve",
        "Os dados possuem uma assimetria negativa (cauda à esquerda) leve.",
        [DistributionSolution.TRANSFORMACAO_QUADRATICA]
    )
    
    ASSIMETRIA_NEGATIVA_FORTE = (
        "Assimetria negativa forte",
        "Os dados possuem uma assimetria negativa (cauda à esquerda) forte.",
        [DistributionSolution.TRANSFORMACAO_REFLECAO_LOG]
    )
    
    CAUDA_PESADA_SEM_ASSIMETRIA = (
        "Cauda pesada sem assimetria significativa",
        "Os dados têm caudas pesadas, mas sem assimetria significativa.",
        [DistributionSolution.WINSORIZATION, DistributionSolution.CENTRALIZACAO_PADRONIZACAO]
    )
    
    CAUDA_PESADA_COM_ASSIMETRIA = (
        "Cauda pesada com assimetria",
        "Os dados têm caudas pesadas e assimetria significativa.",
        [DistributionSolution.TRANSFORMACAO_BOX_COX, DistributionSolution.WINSORIZATION]
    )
    
    DISTRIBUICAO_UNIFORME = (
        "Distribuição aproximadamente uniforme",
        "Os dados seguem uma distribuição aproximadamente uniforme.",
        [DistributionSolution.NENHUMA]
    )
    
    BIMODAL_MULTIMODAL = (
        "Distribuição bimodal ou multimodal",
        "Os dados possuem múltiplos picos, sugerindo a presença de subgrupos.",
        [DistributionSolution.QUANTIZATION]
    )


class DistributionAnalyzer(AnalysisStep):
    """
    Classe para analisar distribuições em dados tabulares e sugerir transformações.
    
    Funcionalidades:
    1. Identifica o tipo de distribuição para cada variável numérica
    2. Calcula estatísticas de forma (assimetria e curtose)
    3. Sugere transformações para aproximar os dados de uma distribuição normal
    4. Fornece diagnósticos detalhados sobre a distribuição dos dados
    """
    
    def __init__(self, min_samples: int = 20):
        """
        Inicializa o analisador de distribuições.
        
        Args:
            min_samples: Número mínimo de amostras necessárias para análise (default: 20)
        """
        self.min_samples = min_samples
    
    def analyze(self, data: pd.DataFrame) -> List[Dict]:
        """
        Analisa a distribuição estatística das variáveis numéricas.
        
        Args:
            data: DataFrame com os dados a serem analisados
            
        Returns:
            Lista de dicionários com resultados para cada coluna
        """
        data = detect_and_replace_placeholders(data)
        results = []
        
        # Selecionar apenas colunas numéricas (não booleanas)
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        
        for col in numeric_cols:
            # Ignorar colunas booleanas (podem estar como 0/1 mas são categóricas)
            if set(data[col].dropna().unique()).issubset({0, 1}) and len(data[col].dropna().unique()) <= 2:
                continue
            
            # Ignorar colunas com muitos zeros ou valores constantes
            if data[col].dropna().nunique() <= 1:
                continue
            
            valid_data = data[col].dropna()
            
            # Ignorar colunas com poucos dados
            if len(valid_data) < 8:
                continue
            
            result = self._analyze_column(valid_data, col)
            results.append(result)
        
        return results
    
    def _analyze_column(self, data: pd.Series, col_name: str) -> Dict[str, Any]:
        """
        Analisa a distribuição de uma coluna numérica.
        
        Args:
            data: Série de dados de uma coluna
            col_name: Nome da coluna sendo analisada
            
        Returns:
            Dicionário com resultados da análise
        """
        # Calcula estatísticas
        statistics = self._calculate_statistics(data)
        
        # Classifica o tipo de distribuição
        dist_type = self._classify_distribution(statistics)
        
        # Obtém ações sugeridas
        actions = self._get_suggested_actions(statistics, dist_type)
        
        # Escolhe a solução principal para destacar (primeira da lista ou NENHUMA)
        solution = actions[0] if actions else DistributionSolution.NENHUMA.name
        
        # Cria descrição textual
        pattern_name, pattern_description = self._identify_scenario(statistics)
        
        # Formata o resultado final conforme o padrão do sistema
        result = {
            'column': col_name,
            'problem': dist_type.name,
            'problem_description': dist_type.value,
            'description': dist_type.value,  # Adicionando alias para compatibilidade com testes
            'solution': solution,
            'actions': actions,
            'statistics': statistics
        }
        
        return result
    
    def _calculate_statistics(self, data: pd.Series) -> Dict[str, float]:
        """
        Calcula estatísticas descritivas para a distribuição.
        
        Args:
            data: Série pandas com os dados 
            
        Returns:
            Dicionário com estatísticas calculadas
        """
        # Garantir que estamos trabalhando com valores numéricos
        if data.dtype == bool:
            data = data.astype(int)
        
        # Estatísticas básicas
        mean = data.mean()
        median = data.median()
        min_val = data.min()
        max_val = data.max()
        std = data.std()
        
        # Estatísticas de forma - usamos método Fisher-Pearson para skewness
        try:
            skewness = float(stats.skew(data, bias=False))
            kurtosis = float(stats.kurtosis(data, fisher=True))
        except (TypeError, ValueError):
            # Falha no cálculo (pode acontecer com dados constantes ou booleanos)
            skewness = 0.0
            kurtosis = 0.0
        
        # Testes de normalidade - usar apenas amostra se dataset for grande
        sample = data.sample(min(1000, len(data)))
        try:
            shapiro_stat, shapiro_p = stats.shapiro(sample)
            ks_stat, ks_p = stats.kstest(stats.zscore(sample), 'norm')
        except (TypeError, ValueError):
            shapiro_p = 0.0
            ks_p = 0.0
        
        # Percentis
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        
        # Mediana/Média (indicador de assimetria)
        median_mean_ratio = median / mean if mean != 0 else float('nan')
        
        return {
            'mean': mean,
            'median': median,
            'min': min_val,
            'max': max_val,
            'std': std,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'shapiro_p': shapiro_p,
            'ks_p': ks_p,
            'q1': q1,
            'q3': q3,
            'iqr': iqr,
            'median_mean_ratio': median_mean_ratio
        }
    
    def _classify_distribution(self, stats: Dict[str, float]) -> DistributionType:
        """
        Classifica o tipo de distribuição com base nas estatísticas.
        
        Args:
            stats: Dicionário com estatísticas da distribuição
            
        Returns:
            Enum DistributionType indicando o tipo de distribuição
        """
        skewness = stats['skewness']
        kurtosis = stats['kurtosis']
        shapiro_p = stats['shapiro_p']
        
        # Verificação de normalidade (usando teste de Shapiro-Wilk e estatísticas de forma)
        is_normal = (abs(skewness) < 0.5 and abs(kurtosis) < 0.5) or shapiro_p > 0.05
        
        if is_normal:
            return DistributionType.NORMAL
        
        # Verificação de uniformidade
        if abs(skewness) < 0.2 and kurtosis < -0.5:
            return DistributionType.UNIFORME
        
        # Verificação de assimetria
        if skewness > 0.5:
            # Verifica se é log-normal
            if skewness > 1.5 and kurtosis > 3:
                return DistributionType.LOGNORMAL
            
            # Verifica se é exponencial
            if abs(skewness - 2) < 0.5 and abs(kurtosis - 6) < 1.5:
                return DistributionType.EXPONENCIAL
                
            return DistributionType.ASSIMETRICA_POSITIVA
            
        if skewness < -0.5:
            return DistributionType.ASSIMETRICA_NEGATIVA
        
        # Verificação de cauda pesada (mediante kurtosis)
        if kurtosis > 3:
            return DistributionType.CAUDA_PESADA
        
        # Padrão caso não seja possível classificar com certeza
        return DistributionType.DESCONHECIDA
    
    def _identify_scenario(self, stats: Dict[str, float]) -> Tuple[str, str]:
        """
        Identifica o cenário de distribuição e retorna sua descrição.
        
        Args:
            stats: Dicionário com estatísticas da distribuição
            
        Returns:
            Tupla com (nome do cenário, descrição detalhada)
        """
        skewness = stats['skewness']
        kurtosis = stats['kurtosis']
        
        # Verifica normalidade
        if abs(skewness) < 0.5 and abs(kurtosis) < 0.5:
            return "Normal", "Os dados seguem uma distribuição aproximadamente normal."
        
        # Verifica assimetria positiva
        if skewness > 0.5:
            if skewness > 2:
                return "Assimetria positiva forte", "Os dados possuem forte assimetria positiva (cauda longa à direita)."
            elif skewness > 1:
                return "Assimetria positiva moderada", "Os dados possuem assimetria positiva moderada."
            else:
                return "Assimetria positiva leve", "Os dados possuem assimetria positiva leve."
        
        # Verifica assimetria negativa
        if skewness < -0.5:
            if skewness < -2:
                return "Assimetria negativa forte", "Os dados possuem forte assimetria negativa (cauda longa à esquerda)."
            else:
                return "Assimetria negativa leve/moderada", "Os dados possuem assimetria negativa."
        
        # Verifica caudas pesadas
        if kurtosis > 3:
            return "Cauda pesada", "Os dados possuem caudas pesadas (mais valores extremos que a distribuição normal)."
        
        # Verifica se é uniforme
        if abs(skewness) < 0.2 and kurtosis < -0.5:
            return "Uniforme", "Os dados seguem uma distribuição aproximadamente uniforme."
        
        # Padrão para outros casos
        return "Não classificada", "Distribuição não classificada claramente em nenhum padrão conhecido."
    
    
    def _get_suggested_actions(self, stats: Dict[str, float], dist_type: DistributionType) -> List[str]:
        """
        Determina as ações sugeridas com base no tipo de distribuição.
        
        Args:
            stats: Dicionário com estatísticas da distribuição
            dist_type: Tipo de distribuição identificado
            
        Returns:
            Lista de strings com nomes das ações sugeridas
        """
        skewness = stats['skewness']
        actions = []
        
        # Para distribuição normal, não são necessárias transformações
        if dist_type == DistributionType.NORMAL:
            return []
        
        # Para distribuição uniforme, geralmente não são necessárias transformações
        if dist_type == DistributionType.UNIFORME:
            return [DistributionSolution.NENHUMA.name]
        
        # Para assimetria positiva ou log-normal
        if dist_type in [DistributionType.ASSIMETRICA_POSITIVA, DistributionType.LOGNORMAL, DistributionType.EXPONENCIAL]:
            # Assimetria forte
            if skewness > 2:
                actions.append(DistributionSolution.TRANSFORMACAO_LOG.name)
                actions.append(DistributionSolution.TRANSFORMACAO_BOX_COX.name)
            # Assimetria moderada
            elif skewness > 1:
                actions.append(DistributionSolution.TRANSFORMACAO_LOG.name)
                actions.append(DistributionSolution.TRANSFORMACAO_RAIZ_QUADRADA.name)
            # Assimetria leve
            else:
                actions.append(DistributionSolution.TRANSFORMACAO_RAIZ_QUADRADA.name)
                actions.append(DistributionSolution.CENTRALIZACAO_PADRONIZACAO.name)
        
        # Para assimetria negativa
        if dist_type == DistributionType.ASSIMETRICA_NEGATIVA:
            # Assimetria forte
            if skewness < -2:
                actions.append(DistributionSolution.TRANSFORMACAO_REFLECAO_LOG.name)
            # Assimetria leve/moderada
            else:
                actions.append(DistributionSolution.TRANSFORMACAO_QUADRATICA.name)
                actions.append(DistributionSolution.TRANSFORMACAO_INVERSA.name)
        
        # Para cauda pesada
        if dist_type == DistributionType.CAUDA_PESADA:
            actions.append(DistributionSolution.WINSORIZATION.name)
            actions.append(DistributionSolution.CLIPPING.name)
            actions.append(DistributionSolution.ALGORITMO_ROBUSTO.name)
        
        return actions

