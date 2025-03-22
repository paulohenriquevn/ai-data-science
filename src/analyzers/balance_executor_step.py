from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
import pandas as pd


class BalanceExecutorStep:
    def __init__(self, strategy: str = "SMOTE", target_column: str = "y"):
        self.strategy = strategy
        self.target_column = target_column

    def execute(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.target_column not in data.columns:
            raise ValueError(f"Coluna alvo '{self.target_column}' não encontrada.")

        X = data.drop(columns=[self.target_column])
        y = data[self.target_column]

        if self.strategy == "SMOTE":
            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X, y)
        elif self.strategy == "OVERSAMPLING":
            # Reamostragem da minoria
            df_majority = data[data[self.target_column] == y.value_counts().idxmax()]
            df_minority = data[data[self.target_column] == y.value_counts().idxmin()]
            df_minority_upsampled = resample(
                df_minority,
                replace=True,
                n_samples=len(df_majority),
                random_state=42
            )
            balanced_df = pd.concat([df_majority, df_minority_upsampled])
            return balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        elif self.strategy == "UNDERSAMPLING":
            # Subamostragem da maioria
            df_majority = data[data[self.target_column] == y.value_counts().idxmax()]
            df_minority = data[data[self.target_column] == y.value_counts().idxmin()]
            df_majority_downsampled = resample(
                df_majority,
                replace=False,
                n_samples=len(df_minority),
                random_state=42
            )
            balanced_df = pd.concat([df_majority_downsampled, df_minority])
            return balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        else:
            raise NotImplementedError(f"Estratégia '{self.strategy}' não suportada.")

        # Reconstruir DataFrame balanceado
        balanced_df = pd.DataFrame(X_res, columns=X.columns)
        balanced_df[self.target_column] = y_res
        return balanced_df


# Exemplo de uso:
# executor = BalanceExecutorStep(strategy="SMOTE", target_column="y")
# balanced_df = executor.execute(df)
