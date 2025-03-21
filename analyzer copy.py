# data_quality_system.py
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import geopandas as gpd
from scipy.stats import shapiro, entropy
from scipy.stats import entropy

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

    def classify(self, df: pd.DataFrame) -> dict:
        """Classifica colunas em tipos específicos conforme a tabela original"""
        type_map = {
            'Numerical': df.select_dtypes(include=np.number).columns.tolist(),
            'Categorical': df.select_dtypes(include=['category', 'object']).columns.tolist(),
            'Temporal': df.select_dtypes(include=['datetime64', 'timedelta64']).columns.tolist(),
            'Text': [col for col in df.columns if self.is_text_column(df[col])]
        }
        
        # Refinar categóricas: colunas com baixa cardinalidade
        for col in type_map['Numerical'][:]:
            if df[col].nunique() < 15 and df[col].dtype != bool:
                type_map['Categorical'].append(col)
                type_map['Numerical'].remove(col)
        
        return type_map

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

# Classe base para analisadores
class BaseAnalyzer(Analyzer):
    def __init__(self, var_types: list):
        self.var_types = var_types
        
    def get_relevant_columns(self, df: pd.DataFrame) -> list:
        type_map = DataTypeClassifier().classify(df)
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
    ('Análise de Texto', ['Text']),
    ('Análise de Outliers (IQR)', ['Numerical']),
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
            if (stats['75%'] - stats['25%']) > 2 * stats['50%']:
                issues[col] = 'Possíveis outliers'
        return {'Técnica': 'Resumo Estatístico', 'Problemas': issues}

class MissingValuesAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__(['Numerical', 'Categorical', 'Temporal', 'Text'])
    
    def analyze(self, X: pd.DataFrame) -> dict:
        missing = X.isna().mean()
        return {'Técnica': 'Análise de Valores Faltantes', 'Problemas': missing[missing > 0].to_dict()}

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
            ent = entropy(pd.Series(X[col]).value_counts(normalize=True))
            if unique_count < self.unique_threshold and ent < 1.0:  # Entropia baixa indica agrupamento de valores
                issues[col] = f'Possível discretização ({unique_count} valores únicos, entropia={ent:.2f})'
        return {'Técnica': 'Análise de Discretização', 'Problemas': issues}


class ZScoreOutlierAnalyzer(BaseAnalyzer):
    def __init__(self, threshold=3):
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

class IsolationForestOutlierAnalyzer(BaseAnalyzer):
    def __init__(self, contamination="auto"):
        super().__init__(['Numerical'])
        self.contamination = contamination

    def analyze(self, X: pd.DataFrame) -> dict:
        issues = {}
        for col in self.get_relevant_columns(X):
            model = IsolationForest(contamination=self.contamination)
            outliers = model.fit_predict(X[[col]].dropna()) == -1
            if outliers.any():
                issues[col] = f"{outliers.sum()} outliers detectados (Isolation Forest)"
        return {'Técnica': 'Análise de Outliers (Isolation Forest)', 'Problemas': issues}

class DBSCANOutlierAnalyzer(BaseAnalyzer):
    def __init__(self, eps=0.5, min_samples=5):
        super().__init__(['Numerical'])
        self.eps = eps
        self.min_samples = min_samples

    def analyze(self, X: pd.DataFrame) -> dict:
        issues = {}
        for col in self.get_relevant_columns(X):
            model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
            outliers = model.fit_predict(X[[col]].dropna()) == -1
            if outliers.any():
                issues[col] = f"{outliers.sum()} outliers detectados (DBSCAN)"
        return {'Técnica': 'Análise de Outliers (DBSCAN)', 'Problemas': issues}

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

class TextAnalysisAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__(['Text'])
    
    def analyze(self, X: pd.DataFrame) -> dict:
        issues = {}
        for col in self.get_relevant_columns(X):
            vectorizer = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS)
            try:
                tfidf_matrix = vectorizer.fit_transform(X[col].dropna())
                if len(vectorizer.get_feature_names_out()) < 5:
                    issues[col] = 'Baixa diversidade textual'
            except:
                issues[col] = 'Erro no processamento de texto'
        return {'Técnica': 'Análise de Texto (TF-IDF)', 'Problemas': issues}

class DataVisualizer:
    def plot_distribution(self, column):
        sns.histplot(column)
        plt.show()

    def plot_outliers(self, column, outliers):
        px.scatter(x=column.index, y=column, color=outliers).show()

class GeospatialAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__(['Geospatial'])
    
    def analyze(self, X: pd.DataFrame) -> dict:
        issues = {}
        if 'geometry' in X.columns:
            gdf = gpd.GeoDataFrame(X)
            gdf.plot()
            plt.show()
        else:
            issues['Geospatial'] = 'Coluna de geometria não encontrada'
        return {'Técnica': 'Análise Geoespacial', 'Problemas': issues}


class TimeSeriesAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__(['Temporal'])
    
    def analyze(self, X: pd.DataFrame) -> dict:
        issues = {}
        for col in self.get_relevant_columns(X):
            try:
                # Converter para DatetimeIndex se necessário
                series = X[col].dropna()
                if not isinstance(series.index, pd.DatetimeIndex):
                    series.index = pd.to_datetime(series.index)
                
                # Definir uma frequência padrão (diária) se não estiver definida
                if series.index.freq is None:
                    series = series.asfreq('D')
                
                # Realizar a decomposição
                decomposition = self.decompose(series)
                decomposition.plot()
                plt.show()
            except Exception as e:
                issues[col] = f"Erro na decomposição da série temporal: {str(e)}"
        return {'Técnica': 'Decomposição de Série Temporal', 'Problemas': issues}

    def decompose(self, series):
        from statsmodels.tsa.seasonal import seasonal_decompose
        return seasonal_decompose(series, model='additive', period=None)

class IQRAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__(['Numerical'])
    
    def analyze(self, X: pd.DataFrame) -> dict:
        issues = {}
        for col in self.get_relevant_columns(X):
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_mask = (X[col] < (Q1 - 1.5 * IQR)) | (X[col] > (Q3 + 1.5 * IQR))
            outlier_count = outlier_mask.sum()
            if outlier_count > 0:
                issues[col] = f"{outlier_count} outliers detectados (IQR)"
        return {'Técnica': 'Análise de Outliers (IQR)', 'Problemas': issues}


class NormalityAnalyzer(BaseAnalyzer):
    def __init__(self):
        super().__init__(['Numerical'])
    
    def analyze(self, X: pd.DataFrame) -> dict:
        issues = {}
        for col in self.get_relevant_columns(X):
            p_value = shapiro(X[col].dropna())[1]  # Teste de Shapiro-Wilk
            if p_value < 0.05:
                issues[col] = f'Distribuição não normal (p-valor={p_value:.3f})'
        return {'Técnica': 'Teste de Normalidade', 'Problemas': issues}


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
            IsolationForestOutlierAnalyzer(),
            DBSCANOutlierAnalyzer(),
            TimeSeriesAnalyzer(),
            IQRAnalyzer(),
            NormalityAnalyzer()
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
            IsolationForestOutlierAnalyzer(),
            DBSCANOutlierAnalyzer(),
            TimeSeriesAnalyzer(),
            IQRAnalyzer(),
            NormalityAnalyzer()
        ]
    
    def inspect(self, df: pd.DataFrame) -> pd.DataFrame:
        report = []
        type_map = DataTypeClassifier().classify(df)
        
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

class DataQualityScore:
    def __init__(self, report_df):
        self.report_df = report_df

    def compute_score(self):
        total_issues = sum(len(problema) for problema in self.report_df["Detalhes"])
        num_columns = self.report_df["Colunas Afetadas"].explode().nunique()
        return max(0, 100 - (total_issues / max(1, num_columns) * 10))  # Escala de 0 a 100


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
    quality_score = DataQualityScore(resultados).compute_score()
    print(f"Pontuação de Qualidade dos Dados: {quality_score:.2f}/100")
    # Exibir resultados
    print("\nRelatório Completo de Qualidade de Dados:")
    print(resultados.to_string())
    
def usandoSmartDataInspector():
    # Gerar e analisar dataset
    df = generate_test_dataset()
    
    # Usar SmartDataInspector
    smart_inspector = SmartDataInspector()
    smart_results = smart_inspector.inspect(df)
    quality_score = DataQualityScore(smart_results).compute_score()
    print(f"Pontuação de Qualidade dos Dados: {quality_score:.2f}/100")
    
    print("\nRelatório Inteligente de Qualidade de Dados:")
    print(smart_results.to_string())    

# Exemplo de uso atualizado
if __name__ == "__main__":
    usandoSmartDataInspector()