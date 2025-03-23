import pandas as pd

class ResidualCollinearityExecutor:
    def __init__(self):
        pass

    def execute(self, df: pd.DataFrame, plan: list) -> pd.DataFrame:
        df = df.copy()

        for step in plan:
            col = step["column"]
            action = step["suggestion"]

            if col not in df.columns:
                continue

            if action in ["REMOVER_VARIAVEL", "REMOVER_REDUNDANTE", "EXCLUIR_POR_RELEVANCIA_BAIXA"]:
                df.drop(columns=[col], inplace=True, errors="ignore")

            elif action == "NENHUMA":
                continue

            else:
                raise NotImplementedError(f"Ação '{action}' não reconhecida no executor de colinearidade residual.")

        return df

# Exemplo de uso:
# executor = ResidualCollinearityExecutor()
# df_final = executor.execute(df, collinearity_plan)
# df_final.head()
