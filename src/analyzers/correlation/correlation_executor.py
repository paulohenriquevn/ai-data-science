import pandas as pd

class CorrelationExecutor:
    def __init__(self):
        pass

    def execute(self, df: pd.DataFrame, plan: list) -> pd.DataFrame:
        df = df.copy()

        for step in plan:
            col = step.get("column")
            action = step.get("suggestion")

            if col not in df.columns:
                continue

            if action == "EXCLUIR_COLUNA":
                df.drop(columns=[col], inplace=True)

            elif action in ["PCA", "REGULARIZATION", "STEPWISE_SELECTION", "NENHUMA"]:
                # Essas ações não modificam diretamente o DataFrame neste ponto
                continue

            else:
                raise NotImplementedError(f"Ação '{action}' não reconhecida no executor de correlação.")

        return df

