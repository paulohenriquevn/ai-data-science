# src/eda_analyzer.py
from enum import Enum
from abc import ABC, abstractmethod
import pandas as pd

class Outliers(Enum):
    IQR_10 = "IQR_10"
    IQR_20 = "IQR_20"
    IQR_30 = "IQR_30"
    Z_SCORE_3 = "Z_SCORE_3"
    Z_SCORE_5 = "Z_SCORE_5"
    Z_SCORE_10 = "Z_SCORE_10"
    IMPACTO_MODERADO = "IMPACTO_MODERADO"
    IMPACTO_FORTE = "IMPACTO_FORTE"
    IMPACTO_EXTREMO = "IMPACTO_EXTREMO"
    NO_OUTLIERS = "NO_OUTLIERS"


class Distributions(Enum):
    NORMAL = "NORMAL"
    LOG_NORMAL = "LOG_NORMAL"
    POISSON = "POISSON"
    BINOMIAL = "BINOMIAL"
    MULTINOMIAL = "MULTINOMIAL"
    EXPONENTIAL = "EXPONENTIAL"
    UNIFORM = "UNIFORM"
    NORMAL_COMPOSTA = "NORMAL_COMPOSTA"   


class Corruption(Enum):
    CORRECT = "CORRECT"
    NO_CORRECT = "NO_CORRECT"
    NO_CORRUPTION = "NO_CORRUPTION"


class Duplicates(Enum):
    REMOVE = "REMOVE"
    AGGREGATE = "AGGREGATE"
    NO_DUPLICATES = "NO_DUPLICATES"


class Balance(Enum):
    BALANCE_MAIOR_80_UNDERSAMPLING = "BALANCE_MAIOR_80_UNDERSAMPLING"
    BALANCE_MAIOR_80_ENSEMBLE = "BALANCE_ENTRE_80_ENSEMBLE"
    BALANCE_MENOR_30_SMOTE = "BALANCE_MENOR_30_SMOTE"
    BALANCE_MENOR_30_OVERSAMPLING = "BALANCE_MENOR_30_OVERSAMPLING"
    BALANCE_MENOR_30_ENSMOTE = "BALANCE_MENOR_30_ENSMOTE"
    BALANCE_MENOR_30_ENSMOTE_TOMEK = "BALANCE_MENOR_30_ENSMOTE_TOMEK"
    NO_BALANCE = "NO_BALANCE"


class Correlation(Enum):    
    CORRELATION_HIGH = "CORRELATION_HIGH"
    CORRELATION_MEDIUM = "CORRELATION_MEDIUM"
    CORRELATION_LOW = "CORRELATION_LOW"
    NO_CORRELATION = "NO_CORRELATION"


class Normalization(Enum):  
    MIN_MAX_SCALER = "MIN_MAX_SCALER"
    STANDARD_SCALER = "STANDARD_SCALER"
    ROBUST_SCALER = "ROBUST_SCALER"  


class CategoricalEncoding(Enum):  
    ORDINAL_ENCODING = "ORDINAL_ENCODING"
    ONE_HOT_ENCODING = "ONE_HOT_ENCODING"
    LABEL_ENCODING = "LABEL_ENCODING"
    TARGET_ENCODING = "TARGET_ENCODING"
    HASHING_TRICK = "HASHING_TRICK" 


class TemporalFeatures(Enum):
    SAZONALITY = "SAZONALITY"
    TRENDS = "TRENDS"
    NO_TEMPORAL_FEATURES = "NO_TEMPORAL_FEATURES"


class InteractionFeatures(Enum):
    POLYNOMIAL_FEATURES = "POLYNOMIAL_FEATURES"
    MULTIPLICATIVE_FEATURES = "MULTIPLICATIVE_FEATURES"
    NO_INTERACTION_FEATURES = "NO_INTERACTION_FEATURES"


class Binning(Enum):
    QUANTILE_BINNING = "QUANTILE_BINNING"
    EQUAL_WIDTH_BINNING = "EQUAL_WIDTH_BINNING"
    NO_BINNING = "NO_BINNING"


class Discretization(Enum):
    DISCRETIZATION_EQUAL_WIDTH = "DISCRETIZATION_EQUAL_WIDTH"
    DISCRETIZATION_QUANTILE = "DISCRETIZATION_QUANTILE"
    NO_DISCRETIZATION = "NO_DISCRETIZATION"


class DimensionalityReduction(Enum):
    LDA = "LDA"
    PCA = "PCA"
    UMAP = "UMAP"
    SELECTION_BY_IMPORTANCE = "SELECTION_BY_IMPORTANCE"
    NO_DIMENSIONALITY_REDUCTION = "NO_DIMENSIONALITY_REDUCTION"


class AnalysisStep(ABC):
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> dict:
        pass





class OutliersAnalyzer(AnalysisStep):
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> dict:
        pass
    
    @abstractmethod
    def report(self) -> Outliers:
        pass


class DistributionAnalyzer(AnalysisStep):
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> dict:
        pass
    
    @abstractmethod
    def report(self) -> Distributions:
        pass


class CorruptionAnalyzer(AnalysisStep):
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> dict:
        pass
    
    @abstractmethod
    def report(self) -> Corruption:
        pass


class DuplicatesAnalyzer(AnalysisStep):
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> dict:
        pass
    
    @abstractmethod
    def report(self) -> Duplicates:
        pass


class BalanceAnalyzer(AnalysisStep):
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> dict:
        pass
    
    @abstractmethod
    def report(self) -> Balance:
        pass


class CorrelationAnalyzer(AnalysisStep):
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> dict:
        pass
    
    @abstractmethod
    def report(self) -> Correlation:
        pass
    

class NormalizationAnalyzer(AnalysisStep):
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> dict:
        pass
    
    @abstractmethod
    def report(self) -> Normalization:
        pass


class CategoricalEncodingAnalyzer(AnalysisStep):
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> dict:
        pass

    @abstractmethod
    def report(self) -> CategoricalEncoding:
        pass
    


class TemporalFeaturesAnalyzer(AnalysisStep):
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> dict:
        pass
    
    @abstractmethod
    def report(self) -> TemporalFeatures:
        pass
    

class InteractionFeaturesAnalyzer(AnalysisStep):
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> dict:
        pass
    
    @abstractmethod
    def report(self) -> InteractionFeatures:
        pass


class BinningAnalyzer(AnalysisStep):
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> dict:
        pass

    @abstractmethod
    def report(self) -> Binning:
        pass


class DiscretizationAnalyzer(AnalysisStep):
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> dict:
        pass

    @abstractmethod
    def report(self) -> Discretization:
        pass


class DimensionalityReductionAnalyzer(AnalysisStep):
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> dict:
        pass
    
    @abstractmethod
    def report(self) -> DimensionalityReduction:
        pass