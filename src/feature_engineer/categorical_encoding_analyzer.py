from enum import Enum
import pandas as pd
from src.feature_engineer.feature_engineer_step import FeatureEngineerStep

class CategoricalEncoding(Enum):  
    ORDINAL_ENCODING = "ORDINAL_ENCODING"
    ONE_HOT_ENCODING = "ONE_HOT_ENCODING"
    LABEL_ENCODING = "LABEL_ENCODING"
    TARGET_ENCODING = "TARGET_ENCODING"
    HASHING_TRICK = "HASHING_TRICK" 
    
class CategoricalEncodingSolution(Enum):
    pass

class CategoricalEncodingAnalyzer(FeatureEngineerStep):
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> dict:
        pass