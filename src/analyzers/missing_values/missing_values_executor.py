import pandas as pd
import numpy as np
from src.analyzers.base.analysis_base import ExecutionStep
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

class MissingValuesExecutor(ExecutionStep):
    def __init__(self, fill_value: float = -999):
        self.fill_value = fill_value

    def execute(self, data: pd.DataFrame, plan: list) -> pd.DataFrame:
        df = data.copy()

        knn_targets = []
        mice_targets = []

        for step in plan:
            col = step.get("column")
            action = step.get("suggestion")

            if col not in df.columns or action == "NENHUMA_ACAO_NECESSARIA":
                continue

            if action == "IMPUTE_MEDIA":
                df[col] = df[col].fillna(df[col].mean())

            elif action == "IMPUTE_MEDIANA":
                df[col] = df[col].fillna(df[col].median())

            elif action == "CATEGORIA_DESCONHECIDA":
                if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
                    df[col] = df[col].fillna("Desconhecido")
                else:
                    df[col] = df[col].fillna(self.fill_value)

            elif action == "ADICIONAR_FLAG_E_IMPUTAR":
                df[f"{col}_missing"] = df[col].isna().astype(int)
                df[col] = df[col].fillna(self.fill_value)

            elif action == "EXCLUIR_LINHAS":
                df = df[df[col].notna()]

            elif action == "EXCLUIR_COLUNA":
                df.drop(columns=[col], inplace=True)

            elif action == "KNN_IMPUTER":
                knn_targets.append(col)

            elif action == "MICE_IMPUTER":
                mice_targets.append(col)

            else:
                raise NotImplementedError(f"Ação de imputação '{action}' não reconhecida.")

        # Aplicar KNN Imputer
        if knn_targets:
            imputer = KNNImputer()
            knn_data = df[knn_targets]
            imputed = imputer.fit_transform(knn_data)
            df[knn_targets] = imputed

        # Aplicar MICE Imputer
        if mice_targets:
            imputer = IterativeImputer(random_state=42)
            mice_data = df[mice_targets]
            imputed = imputer.fit_transform(mice_data)
            df[mice_targets] = imputed

        return df
