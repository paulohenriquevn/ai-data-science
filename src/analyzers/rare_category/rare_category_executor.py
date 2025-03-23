import pandas as pd


class RareCategoryExecutor:
    def __init__(self, rare_threshold: float = 0.01):
        self.rare_threshold = rare_threshold

    def execute(self, df: pd.DataFrame, plan: list) -> pd.DataFrame:
        df = df.copy()

        for step in plan:
            col = step["column"]
            action = step["suggestion"]

            if col not in df.columns or df[col].isnull().all():
                continue

            if action == "AGRUPAR_RARAS":
                freq = df[col].value_counts(normalize=True)
                rare_categories = freq[freq < self.rare_threshold].index.tolist()
                df[col] = df[col].apply(lambda x: "OUTROS" if x in rare_categories else x)

            elif action == "REDUZIR_CARDINALIDADE":
                # Estratégia básica: manter as N categorias mais frequentes, agrupar o resto
                top_n = 10
                top_categories = df[col].value_counts().nlargest(top_n).index
                df[col] = df[col].apply(lambda x: x if x in top_categories else "OUTROS")

            elif action == "NENHUMA":
                continue

            else:
                raise NotImplementedError(f"Ação '{action}' não reconhecida para rare categories.")

        return df
