import pandas as pd
import numpy as np


class FeatureInteractionExecutor:
    def __init__(self):
        pass

    def execute(self, df: pd.DataFrame, plan: list) -> pd.DataFrame:
        df = df.copy()

        for step in plan:
            col1, col2 = step['column']
            action = step['suggestion']

            if col1 not in df.columns or col2 not in df.columns:
                continue

            try:
                if action == "CRIAR_INTERACAO_PRODUTO":
                    new_col = f"{col1}__mul__{col2}"
                    df[new_col] = df[col1] * df[col2]

                elif action == "CRIAR_RAZAO":
                    new_col = f"{col1}__div__{col2}"
                    df[new_col] = df[col1] / df[col2].replace(0, np.nan)

                elif action == "CRIAR_DIFERENCA":
                    new_col = f"{col1}__sub__{col2}"
                    df[new_col] = df[col1] - df[col2]

                elif action == "NENHUMA":
                    continue

                else:
                    raise NotImplementedError(f"Ação '{action}' não reconhecida na criação de interações.")

            except Exception as e:
                print(f"[Erro ao gerar interação entre {col1} e {col2}]: {e}")

        return df

