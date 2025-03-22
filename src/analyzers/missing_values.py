from enum import Enum
from typing import List, Dict, Any
import pandas as pd
from scipy import stats
from src.analyzers.analysis_step import AnalysisStep


class MissingValuesProblemType(Enum):
    """Classificação de problemas de dados ausentes"""
    POUCO = "Poucos ausentes (<5%)"
    MUITO = "Muitos ausentes (>20%)"
    ESTRUTURAL = "Ausência estrutural (MCAR)"
    RELACIONADA = "Relacionada a outras variáveis (MAR)"
    RELACIONADA_AO_VALOR = "Relacionada ao próprio valor (MNAR)"
    PADRAO_SISTEMATICO = "Padrão sistemático"
    ALVO_MISSING = "Variável alvo com ausentes"
    NENHUM = "Nenhum problema de ausência de dados"
    


class MissingValuesSolution(Enum):
    """Técnicas de tratamento para valores ausentes"""
    EXCLUIR_LINHAS = "Excluir linhas afetadas"
    IMPUTE_MEDIA = "Imputação por média"
    IMPUTE_MEDIANA = "Imputação por mediana"
    IMPUTE_MODA = "Imputação por moda"
    KNN_IMPUTER = "Imputação com KNN"
    MICE_IMPUTER = "Imputação múltipla (MICE)"
    ADICIONAR_FLAG_E_IMPUTAR = "Adicionar flag e imputar"
    TRATAMENTO_COM_MODELOS = "Tratamento com modelos (ex: XGBoost)"
    CATEGORIA_DESCONHECIDA = "Categoria 'Desconhecido'"
    NENHUMA_ACAO_NECESSARIA = "Nenhuma ação necessária"


class MissingValuesAnalyzer(AnalysisStep):
    """
    Classe completa para análise e sugestão de tratamento de valores ausentes em dados tabulares.
    
    Funcionalidades:
    1. Calcula estatísticas de valores ausentes
    2. Classifica o tipo de problema (MCAR, MAR, MNAR)
    3. Detecta padrões sistemáticos
    4. Sugere técnicas de tratamento específicas
    Uso:
    analyzer = MissingValuesAnalyzer()
    results = analyzer.analyze(df)
    """
    
    SOLUTION_MAP = {
        MissingValuesProblemType.NENHUM: [
            MissingValuesSolution.NENHUMA_ACAO_NECESSARIA
        ],
        MissingValuesProblemType.POUCO: [
            MissingValuesSolution.EXCLUIR_LINHAS,
            MissingValuesSolution.IMPUTE_MEDIA,
            MissingValuesSolution.IMPUTE_MEDIANA,
            MissingValuesSolution.KNN_IMPUTER
        ],
        MissingValuesProblemType.MUITO: [
            MissingValuesSolution.MICE_IMPUTER,
            MissingValuesSolution.TRATAMENTO_COM_MODELOS
        ],
        MissingValuesProblemType.ESTRUTURAL: [
            MissingValuesSolution.EXCLUIR_LINHAS,
            MissingValuesSolution.IMPUTE_MEDIA
        ],
        MissingValuesProblemType.RELACIONADA: [
            MissingValuesSolution.MICE_IMPUTER,
            MissingValuesSolution.ADICIONAR_FLAG_E_IMPUTAR
        ],
        MissingValuesProblemType.RELACIONADA_AO_VALOR: [
            MissingValuesSolution.ADICIONAR_FLAG_E_IMPUTAR,
            MissingValuesSolution.TRATAMENTO_COM_MODELOS
        ],
        MissingValuesProblemType.PADRAO_SISTEMATICO: [
            MissingValuesSolution.ADICIONAR_FLAG_E_IMPUTAR,
            MissingValuesSolution.TRATAMENTO_COM_MODELOS
        ],
        MissingValuesProblemType.ALVO_MISSING: [
            MissingValuesSolution.EXCLUIR_LINHAS,
            MissingValuesSolution.TRATAMENTO_COM_MODELOS
        ]
    }

    def analyze(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Executa a análise completa para todas as colunas do DataFrame.
        
        Args:
            data (pd.DataFrame): DataFrame a ser analisado
            
        Returns:
            List[Dict]: Lista de dicionários com resultados detalhados por coluna
        """
        results = []
        missing_stats = self._calculate_missing_stats(data)
        
        for col in data.columns:
            result = self._analyze_column(data, col, missing_stats[col])
            results.append(result)
            
        return results

    def _analyze_column(self, data: pd.DataFrame, col: str, stats: Dict) -> Dict[str, Any]:
        """Analisa uma coluna individual"""
        result = {
            'column': col,
            'problem': MissingValuesProblemType.NENHUM.name,
            'description': MissingValuesProblemType.NENHUM.value,
            'statistics': stats,
            'actions': [MissingValuesSolution.NENHUMA_ACAO_NECESSARIA.name]
        }
        
        if stats['missing_count'] > 0:
            problem_type = self._classify_problem(data, col, stats)
            result.update({
                'problem': problem_type.name,
                'description': problem_type.value,
                'actions': [s.name for s in self.SOLUTION_MAP[problem_type]]
            })
            
            # Sugestão especial para categóricos
            if pd.api.types.is_object_dtype(data[col]) and problem_type != MissingValuesProblemType.NENHUM:
                result['actions'].append(MissingValuesSolution.CATEGORIA_DESCONHECIDA.name)
        
        # Adicionar a sugestão principal
        result['suggestion'] = self._get_primary_suggestion(result['problem'], data[col].dtype)
        
        return result

    def _get_primary_suggestion(self, problem_type_name: str, dtype) -> MissingValuesSolution:
        """
        Determina a sugestão principal com base no tipo de problema e tipo de dados.
        
        Args:
            problem_type_name: Nome do tipo de problema
            dtype: Tipo de dados da coluna
            
        Returns:
            MissingValuesSolution mais apropriada para o caso
        """
        problem_type = MissingValuesProblemType[problem_type_name]
        
        if problem_type == MissingValuesProblemType.NENHUM:
            return MissingValuesSolution.NENHUMA_ACAO_NECESSARIA
        
        # Para colunas categóricas (object ou category)
        if pd.api.types.is_object_dtype(dtype) or isinstance(dtype, pd.CategoricalDtype):
            return MissingValuesSolution.CATEGORIA_DESCONHECIDA
                
        if problem_type == MissingValuesProblemType.POUCO:
            return MissingValuesSolution.IMPUTE_MEDIA
        
        if problem_type == MissingValuesProblemType.MUITO:
            return MissingValuesSolution.MICE_IMPUTER
        
        if problem_type == MissingValuesProblemType.ESTRUTURAL:
            return MissingValuesSolution.IMPUTE_MEDIA
        
        if problem_type == MissingValuesProblemType.RELACIONADA:
            return MissingValuesSolution.MICE_IMPUTER
        
        if problem_type == MissingValuesProblemType.RELACIONADA_AO_VALOR:
            return MissingValuesSolution.ADICIONAR_FLAG_E_IMPUTAR
        
        if problem_type == MissingValuesProblemType.PADRAO_SISTEMATICO:
            return MissingValuesSolution.ADICIONAR_FLAG_E_IMPUTAR
        
        if problem_type == MissingValuesProblemType.ALVO_MISSING:
            return MissingValuesSolution.EXCLUIR_LINHAS
        
        # Caso padrão se nenhuma condição for atendida
        return MissingValuesSolution.NENHUMA_ACAO_NECESSARIA


    def _calculate_missing_stats(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """Calcula estatísticas detalhadas de valores ausentes"""
        stats = {}
        for col in data.columns:
            missing = data[col].isna()
            stats[col] = {
                'missing_count': missing.sum(),
                'missing_percent': round(missing.mean() * 100, 2),
                'dtype': str(data[col].dtype),
                'unique_values': data[col].nunique() if pd.api.types.is_object_dtype(data[col]) else None
            }
        return stats

    def _classify_problem(self, data: pd.DataFrame, col: str, stats: Dict) -> MissingValuesProblemType:
        """Classifica o tipo de problema de valores ausentes"""
        missing_percent = stats['missing_percent']
        
        # Sem valores ausentes
        if missing_percent == 0:
            return MissingValuesProblemType.NENHUM
            
        # Verificação de variável alvo - apenas se o nome da coluna contém palavras-chave específicas
        target_keywords = {'target', 'alvo', 'class', 'label', 'y'}
        if any(keyword in col.lower() for keyword in target_keywords):
            return MissingValuesProblemType.ALVO_MISSING
            
        # Classificação por percentual 
        if missing_percent <= 5:
            return MissingValuesProblemType.POUCO
            
        if missing_percent > 20:
            return MissingValuesProblemType.MUITO
        
        # Testes adicionais apenas se estiver entre 5% e 20%
        if self._has_systematic_pattern(data, col):
            return MissingValuesProblemType.PADRAO_SISTEMATICO
            
        if self._is_mar(data, col):
            return MissingValuesProblemType.RELACIONADA
            
        if self._is_mnar(data, col):
            return MissingValuesProblemType.RELACIONADA_AO_VALOR
            
        return MissingValuesProblemType.ESTRUTURAL

    def _is_target_variable(self, col: str, data: pd.DataFrame) -> bool:
        """Verifica se a coluna é a variável target"""
        target_keywords = {'target', 'alvo', 'class', 'label', 'y'}
        return any(keyword in col.lower() for keyword in target_keywords)

    def _is_mar(self, data: pd.DataFrame, col: str) -> bool:
        """Detecta Missing at Random (MAR) usando teste t de Student"""
        for other_col in data.columns.drop(col):
            if pd.api.types.is_numeric_dtype(data[other_col]):
                try:
                    group_present = data.loc[data[col].notna(), other_col]
                    group_missing = data.loc[data[col].isna(), other_col]
                    
                    if len(group_present) < 2 or len(group_missing) < 2:
                        continue
                        
                    _, p_value = stats.ttest_ind(group_present, group_missing, equal_var=False)
                    if p_value < 0.05:
                        return True
                except:
                    continue
        return False

    def _is_mnar(self, data: pd.DataFrame, col: str) -> bool:
        """Detecta Missing Not at Random (MNAR) usando correlação com extremos"""
        if pd.api.types.is_numeric_dtype(data[col]):
            try:
                q1 = data[col].quantile(0.25)
                q3 = data[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5*iqr
                upper_bound = q3 + 1.5*iqr
                
                extreme_values = (data[col] < lower_bound) | (data[col] > upper_bound)
                return data[col].isna().corr(extreme_values, method='kendall') > 0.3
            except:
                return False
        return False

    def _has_systematic_pattern(self, data: pd.DataFrame, col: str) -> bool:
        """Detecta padrões sistemáticos usando autocorrelação"""
        try:
            na_series = data[col].isna().astype(int)
            if len(na_series) > 10:
                return na_series.autocorr(lag=1) > 0.3
        except:
            return False
        return False