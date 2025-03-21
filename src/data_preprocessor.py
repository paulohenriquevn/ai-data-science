# src/data_preprocessor.py
from abc import ABC, abstractmethod
import pandas as pd

class PreprocessingStep(ABC):
    @abstractmethod
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

class DataTypeValidator(PreprocessingStep):
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        pass
    
    def _validate_column_types(self, data: pd.DataFrame) -> None:
        pass

class DuplicateCleaner(PreprocessingStep):
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        pass
    
    def _find_duplicates(self, data: pd.DataFrame) -> pd.DataFrame:
        pass
    
class DataCorruptionHandler(PreprocessingStep):
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        pass
    
    def _detect_invalid_entries(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

class ScalerSelector(PreprocessingStep):
    def __init__(self, strategy: str = "standard"):
        pass
    
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

class CategoricalEncoder(PreprocessingStep):
    def process(self, data: pd.DataFrame) -> pd.DataFrame:
        pass
    
    def _encode_one_hot(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

class DataPreprocessor:
    def __init__(self):
        self.pipeline = [
            DataTypeValidator(),
            DuplicateCleaner(),
        ]
    
    def execute_pipeline(self, data: pd.DataFrame) -> pd.DataFrame:
        pass