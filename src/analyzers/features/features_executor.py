import numpy as np
import pandas as pd

class FeatureEngineeringStep:
    def __init__(self, plan):
        self.plan = plan

    def transform(self, df):
        df = df.copy()

        # Aplicar transformações matemáticas
        for col, method in self.plan.get("transformations", {}).items():
            if col not in df.columns:
                continue
            if method == "sqrt":
                df[f"{col}_sqrt"] = np.sqrt(np.maximum(df[col], 0))
            elif method == "square":
                df[f"{col}_squared"] = np.square(df[col])
            elif method == "log":
                df[f"{col}_log"] = np.log1p(np.maximum(df[col], 0))
            elif method == "inverse":
                df[f"{col}_inverse"] = 1 / (df[col] + 1e-6)

        # Criar flags de ausência
        for col in self.plan.get("flags", []):
            if col in df.columns:
                df[f"flag_{col}_missing"] = df[col].isna().astype(int)

        # Criar interações
        for col1, col2 in self.plan.get("interactions", []):
            if col1 in df.columns and col2 in df.columns:
                df[f"{col1}_x_{col2}"] = df[col1] * df[col2]

        return df
