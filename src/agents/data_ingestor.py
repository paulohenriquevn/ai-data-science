# agents/data_ingestor.py
import pandas as pd
from .base_agent import EDAAgent

class DataIngestor(EDAAgent):
    def analyze(self, file_path: str) -> pd.DataFrame:
        # Implementar lógica de carregamento e pré-processamento
        df = pd.read_csv(file_path)
        df = df.replace(-999, np.nan)
        print(f"Dados com {df.shape[0]} registros e {df.shape[1]} colunas carregados")
        return df