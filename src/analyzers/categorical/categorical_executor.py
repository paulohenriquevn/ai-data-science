import pandas as pd
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

class CategoricalExecutor:
    def __init__(self):
        self.encoders = {}

    def execute(self, df: pd.DataFrame, plan: list) -> pd.DataFrame:
        df = df.copy()

        for step in plan:
            col = step.get("column")
            strategy = step.get("suggestion")

            if col not in df.columns:
                continue

            try:
                if strategy == "ONE_HOT":
                    encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
                    encoded = encoder.fit_transform(df[[col]].fillna("desconhecido"))
                    cols = [f"{col}_{cat}" for cat in encoder.categories_[0]]
                    encoded_df = pd.DataFrame(encoded, columns=cols, index=df.index)
                    df = pd.concat([df.drop(columns=[col]), encoded_df], axis=1)
                    self.encoders[col] = encoder

                elif strategy == "ORDINAL":
                    encoder = OrdinalEncoder()
                    df[col] = encoder.fit_transform(df[[col]].fillna("desconhecido"))
                    self.encoders[col] = encoder

                elif strategy == "NUMERIC_CAST":
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-999)

                elif strategy == "NENHUMA":
                    continue

                else:
                    raise NotImplementedError(f"Estratégia '{strategy}' não reconhecida para a coluna '{col}'.")

            except Exception as e:
                print(f"[Erro ao aplicar '{strategy}' na coluna '{col}']: {e}")

        return df