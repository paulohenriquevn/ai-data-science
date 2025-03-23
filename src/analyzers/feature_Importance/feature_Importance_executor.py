import pandas as pd


class FeatureImportanceExecutor:
    def __init__(self):
        pass

    def execute(self, df: pd.DataFrame, plan: list) -> pd.DataFrame:
        df = df.copy()
        cols_to_remove = []

        for step in plan:
            col = step["column"]
            action = step["suggestion"]

            if col not in df.columns:
                continue

            if action == "REMOVER_VARIAVEL" or action == "LIMITE_TOP_N":
                cols_to_remove.append(col)

            elif action == "NENHUMA":
                continue

            else:
                raise NotImplementedError(f"Ação '{action}' não reconhecida na seleção de variáveis.")

        df.drop(columns=cols_to_remove, inplace=True, errors="ignore")
        return df

# Exemplo de uso:
# executor = FeatureImportanceExecutor()
# df_filtered = executor.execute(df, feature_importance_plan)
# df_filtered.head()
