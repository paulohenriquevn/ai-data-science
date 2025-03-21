# data_quality_system.py
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN

# Interface base para todos os analisadores
class Analyzer(ABC):
    @abstractmethod
    def analyze(self, X: pd.DataFrame) -> dict:
        """Retorna um dict com técnica, variáveis problemáticas e descrição do problema."""
        pass

# Classificador de tipos de dados
class DataTypeClassifier:
    def __init__(self, cardinality_threshold=0.9, avg_length_threshold=20):
        self.cardinality_threshold = cardinality_threshold
        self.avg_length_threshold = avg_length_threshold

    def is_text_column(self, column):
        unique_ratio = column.nunique() / len(column)
        avg_length = column.astype(str).apply(len).mean()
        return unique_ratio > self.cardinality_threshold and avg_length > self.avg_length_threshold
    
    def is_temporal_column(self, column):
        try:
            pd.to_datetime(column)
            return True
        except ValueError:
            return False

    def is_numeric_column(self, column):
        return pd.api.types.is_numeric_dtype(column)
    
    @staticmethod
    def classify(df: pd.DataFrame) -> dict:
        """Classifica colunas em tipos específicos conforme a tabela original"""
        type_map = {
            'Numerical': df.select_dtypes(include=np.number).columns.tolist(),
            'Categorical': [col for col in df.columns if df[col].nunique() < 20 and df[col].dtype == 'object'],
            'Temporal': df.select_dtypes(include=['datetime64', 'timedelta64']).columns.tolist(),
            'Text': [col for col in df.columns if df[col].apply(lambda x: isinstance(x, str) and ' ' in x).any()]
        }
        
        # Refinar categóricas: colunas com baixa cardinalidade
        for col in type_map['Numerical'][:]:
            if df[col].nunique() < 15 and df[col].dtype != bool:
                type_map['Categorical'].append(col)
                type_map['Numerical'].remove(col)
        
        return type_map

# Classe base para analisadores
class BaseAnalyzer(Analyzer):
    def __init__(self, var_types: list):
        self.var_types = var_types
        
    def get_relevant_columns(self, df: pd.DataFrame) -> list:
        type_map = DataTypeClassifier.classify(df)
        return [col for t in self.var_types for col in type_map.get(t, [])]

# Configuração dos analisadores conforme tabela original
analyzers_config = [
    ('Resumo Estatístico', ['Numerical']),
    ('Análise de Valores Faltantes', ['Numerical', 'Categorical', 'Temporal', 'Text']),
    ('Histograma', ['Numerical']),
    ('Gráficos de Barras', ['Categorical']),
    ('Boxplot', ['Numerical']),
    ('Análise de Tendências', ['Temporal']),
    ('Correlação Temporal', ['Temporal']),
    ('Mapa de Correlação', ['Numerical']),
    ('Gráficos de Dispersão', ['Numerical']),
    ('Análise de Outliers', ['Numerical']),
    ('Tabelas de Contingência', ['Categorical']),
    ('Análise de Texto', ['Text'])
]

class AnalyzerFactory:
    @staticmethod
    def create_all():
        analyzers = []
        for name, var_types in analyzers_config:
            analyzers.append(CustomAnalyzer(name, var_types))
        return analyzers

class CustomAnalyzer(BaseAnalyzer):
    def __init__(self, technique_name: str, var_types: list):
        super().__init__(var_types)
        self.technique_name = technique_name
    
    def analyze(self, X: pd.DataFrame) -> dict:
        return {'Técnica': self.technique_name, 'Problemas': {}}


# Implementações concretas dos analisadores
class SummaryStatisticsAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__(['Numerical'])
    
    def analyze(self, X: pd.DataFrame) -> dict:
        issues = {}
        for col in self.get_relevant_columns(X):
            stats = X[col].describe()
            if stats['std'] > 2 * stats['mean']:
                issues[col] = 'Alta variabilidade'
            if (stats['75%'] - stats['25%']) > 1.5 * stats['IQR']:   
                issues[col] = 'Possíveis outliers'
        return {'Técnica': 'Resumo Estatístico', 'Problemas': issues}

class MissingValuesAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__(['Numerical', 'Categorical', 'Temporal', 'Text'])
    
    def analyze(self, X: pd.DataFrame) -> dict:
        missing = X.isna().mean()
        return {'Técnica': 'Análise de Valores Faltantes', 'Problemas': missing[missing > 0].to_dict()}



class IsolationForestAnalyzer:
    def __init__(self, contamination=0.05):
        self.contamination = contamination

    def detect_outliers(self, column):
        model = IsolationForest(contamination=self.contamination)
        return model.fit_predict(column.values.reshape(-1, 1)) == -1


class DBSCANAnalyzer:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples

    def detect_outliers(self, column):
        model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        return model.fit_predict(column.values.reshape(-1, 1)) == -1


class HistogramAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__(['Numerical'])
    
    def analyze(self, X: pd.DataFrame) -> dict:
        issues = {}
        for col in self.get_relevant_columns(X):
            skew = stats.skew(X[col].dropna())
            if abs(skew) > 1.0:
                issues[col] = f'Assimetria significativa (Skewness: {skew:.2f})'
        return {'Técnica': 'Histograma', 'Problemas': issues}

class ClassImbalanceAnalyzer(BaseAnalyzer):
    def __init__(self, threshold=0.8):
        super().__init__(['Categorical'])
        self.threshold = threshold
    
    def analyze(self, X: pd.DataFrame) -> dict:
        issues = {}
        for col in self.get_relevant_columns(X):
            counts = X[col].value_counts(normalize=True)
            if counts.max() > self.threshold:
                issues[col] = f'Desbalanceamento ({counts.idxmax()}: {counts.max():.1%})'
        return {'Técnica': 'Balanceamento de Classes', 'Problemas': issues}

class StationarityAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__(['Temporal'])
    
    def analyze(self, X: pd.DataFrame) -> dict:
        issues = {}
        for col in self.get_relevant_columns(X):
            result = adfuller(X[col].dropna())
            if result[1] > 0.05:
                issues[col] = f'Série não estacionária (p-valor={result[1]:.3f})'
        return {'Técnica': 'Teste de Estacionariedade (ADF)', 'Problemas': issues}

class TemporalCorrelationAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__(['Temporal'])
    
    def analyze(self, X: pd.DataFrame) -> dict:
        issues = {}
        # Implementação simplificada para exemplo
        for col in self.get_relevant_columns(X):
            if len(X[col]) > 1:
                autocorr = X[col].autocorr()
                if abs(autocorr) > 0.7:
                    issues[col] = f'Alta autocorrelação ({autocorr:.2f})'
        return {'Técnica': 'Correlação Temporal', 'Problemas': issues}

class DiscretizationAnalyzer(BaseAnalyzer):
    def __init__(self, unique_threshold=10):
        super().__init__(['Numerical'])
        self.unique_threshold = unique_threshold
        
    def analyze(self, X: pd.DataFrame) -> dict:
        issues = {}
        for col in self.get_relevant_columns(X):
            unique_count = X[col].nunique()
            if unique_count < self.unique_threshold:
                issues[col] = f'Possível discretização ({unique_count} valores únicos)'
        return {'Técnica': 'Análise de Discretização', 'Problemas': issues}

class ZScoreOutlierAnalyzer(BaseAnalyzer):
    def __init__(self, threshold=2.5):
        super().__init__(['Numerical'])
        self.threshold = threshold
        
    def analyze(self, X: pd.DataFrame) -> dict:
        issues = {}
        for col in self.get_relevant_columns(X):
            z_scores = np.abs(stats.zscore(X[col].dropna()))
            outliers = (z_scores > self.threshold).sum()
            if outliers > 0:
                issues[col] = f"{outliers} outliers detectados (Z > {self.threshold})"
        return {'Técnica': 'Análise de Outliers (Z-Score)', 'Problemas': issues}

class TextAnalysisAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__(['Text'])
    
    def analyze(self, X: pd.DataFrame) -> dict:
        issues = {}
        for col in self.get_relevant_columns(X):
            vectorizer = TfidfVectorizer()
            try:
                vectorizer.fit_transform(X[col])
                if len(vectorizer.get_feature_names_out()) < 5:
                    issues[col] = 'Baixa diversidade textual'
            except:
                issues[col] = 'Problema no processamento de texto'
        return {'Técnica': 'Análise de Texto (TF-IDF)', 'Problemas': issues}

# Classe inspector principal
class DataQualityInspector:
    def __init__(self):
        self.analyzers = [
            SummaryStatisticsAnalyzer(),
            MissingValuesAnalyzer(),
            HistogramAnalyzer(),
            ClassImbalanceAnalyzer(),
            StationarityAnalyzer(),
            TemporalCorrelationAnalyzer(),
            DiscretizationAnalyzer(),
            ZScoreOutlierAnalyzer(),
            TextAnalysisAnalyzer()
        ]
    
    def inspect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Executa todas as análises e retorna um relatório consolidado."""
        report = []
        for analyzer in self.analyzers:
            try:
                result = analyzer.analyze(df)
                if result['Problemas']:
                    report.append({
                        'Técnica': result['Técnica'],
                        'Problemas': result['Problemas']
                    })
            except Exception as e:
                print(f"Erro na análise {analyzer.__class__.__name__}: {str(e)}")
        return pd.DataFrame(report)

class SmartDataInspector:
    def __init__(self):
        self.analyzers = AnalyzerFactory.create_all()
        self.analyzers += [
            SummaryStatisticsAnalyzer(),
            MissingValuesAnalyzer(),
            HistogramAnalyzer(),
            ClassImbalanceAnalyzer(),
            StationarityAnalyzer(),
            TemporalCorrelationAnalyzer(),
            DiscretizationAnalyzer(),
            ZScoreOutlierAnalyzer(),
            TextAnalysisAnalyzer()
        ]
    
    def inspect(self, df: pd.DataFrame) -> pd.DataFrame:
        report = []
        type_map = DataTypeClassifier.classify(df)
        
        for analyzer in self.analyzers:
            try:
                result = analyzer.analyze(df)
                if result['Problemas']:
                    report.append({
                        'Técnica': result['Técnica'],
                        'Tipo Variável': ', '.join(analyzer.var_types),
                        'Colunas Afetadas': list(result['Problemas'].keys()),
                        'Detalhes': result['Problemas']
                    })
            except Exception as e:
                print(f"Erro em {analyzer.__class__.__name__}: {str(e)}")
        
        return pd.DataFrame(report)

# Função para gerar dataset de teste corrigida
def generate_test_dataset():
    data = {
        'missing_col': [np.nan if i % 10 == 0 else i for i in range(100)],
        'discretized_col': np.random.randint(1, 5, 100),
        'outlier_col': np.concatenate([np.random.normal(0, 1, 95), np.array([10, -10, 15, -15, 20])]),
        'imbalanced_cat': np.random.choice(['X','Y'], 100, p=[0.9,0.1]),
        'non_stationary_date': pd.date_range("2020-01-01", periods=100) + pd.to_timedelta(np.arange(100)), 
        'low_diversity_text': ['repeat']*90 + ['unique']*10,
        'normal_num': np.random.normal(0, 1, 100),
        'balanced_cat': np.random.choice(['A','B','C'], 100)
    }
    return pd.DataFrame(data)

# Exemplo de uso
def usandoDataQualityInspector():
    # Gerar e analisar dataset
    df = generate_test_dataset()
    inspector = DataQualityInspector()
    resultados = inspector.inspect(df)
    
    # Exibir resultados
    print("\nRelatório Completo de Qualidade de Dados:")
    print(resultados.to_string())
    
def usandoSmartDataInspector():
    # Gerar e analisar dataset
    df = generate_test_dataset()
    
    # Usar SmartDataInspector
    smart_inspector = SmartDataInspector()
    smart_results = smart_inspector.inspect(df)
    
    print("\nRelatório Inteligente de Qualidade de Dados:")
    print(smart_results.to_string())    

# Exemplo de uso atualizado
if __name__ == "__main__":
    usandoSmartDataInspector()