import pandas as pd


class LowVariabilityExecutor:
    def __init__(self):
        pass

    def execute(self, df: pd.DataFrame, plan: list) -> pd.DataFrame:
        df = df.copy()

        for step in plan:
            col = step['column']
            action = step['suggestion']

            if col not in df.columns:
                continue

            if action == 'REMOVER_CONSTANTE':
                df.drop(columns=[col], inplace=True)

            elif action == 'NENHUMA':
                continue

            else:
                raise NotImplementedError(f"Ação '{action}' não reconhecida no executor de baixa variabilidade.")

        return df