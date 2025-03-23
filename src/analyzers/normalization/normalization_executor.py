import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from src.analyzers.base.analysis_base import ExecutionStep


class NormalizationExecutor(ExecutionStep):
    def __init__(self):
        self.scalers = {}

    def execute(self, df: pd.DataFrame, plan: list) -> pd.DataFrame:
        df = df.copy()

        for step in plan:
            col = step['column']
            action = step['suggestion']

            if col not in df.columns:
                continue

            # Redimensiona para formato (n, 1)
            values = df[[col]].values

            if action == 'STANDARD_SCALER':
                scaler = StandardScaler()
            elif action == 'ROBUST_SCALER':
                scaler = RobustScaler()
            elif action == 'NENHUMA':
                continue
            else:
                raise NotImplementedError(f"Ação de scaling '{action}' não reconhecida.")

            try:
                df[col] = scaler.fit_transform(values)
                self.scalers[col] = scaler
            except Exception as e:
                print(f"[Erro ao aplicar {action} em {col}]: {e}")

        return df


