import pandas as pd
import numpy as np

class MissingValuesExecutor:
    def __init__(self, plan: dict, fill_value: float = -999):
        self.plan = plan
        self.fill_value = fill_value

    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        for col, strategy in self.plan.items():
            if strategy == "NENHUMA_ACAO_NECESSARIA":
                continue
            elif strategy == "IMPUTE_MEDIA":
                df[col] = df[col].fillna(df[col].mean())
            elif strategy == "IMPUTE_MEDIANA":
                df[col] = df[col].fillna(df[col].median())
            elif strategy == "CATEGORIA_DESCONHECIDA":
                if df[col].dtype == object:
                    df[col] = df[col].fillna("Desconhecido")
                else:
                    df[col] = df[col].fillna(self.fill_value)
            elif strategy == "ADICIONAR_FLAG_E_IMPUTAR":
                df[f"{col}_missing"] = df[col].isna().astype(int)
                df[col] = df[col].fillna(self.fill_value)
            elif strategy in ["KNN_IMPUTER", "MICE_IMPUTER"]:
                # Marcar como ação futura de imputação multivariada
                df[col] = df[col]  # Nenhuma ação por enquanto
            else:
                raise NotImplementedError(f"Estratégia de imputação '{strategy}' não implementada.")
        return df
