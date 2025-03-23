import pandas as pd
import numpy as np

class OutlierExecutor:
    def __init__(self):
        pass

    def execute(self, df: pd.DataFrame, plan: list) -> pd.DataFrame:
        df = df.copy()

        for step in plan:
            col = step['column']
            action = step['suggestion']
            params = step['parameters']

            if col not in df.columns:
                continue

            if action == 'WINSORIZATION':
                lower = params['q1'] - 1.5 * params['iqr']
                upper = params['q3'] + 1.5 * params['iqr']
                df[col] = df[col].clip(lower, upper)

            elif action == 'CLIPPING':
                # Aplica um clipping mais leve (ex: 1º e 99º percentis)
                lower = df[col].quantile(0.01)
                upper = df[col].quantile(0.99)
                df[col] = df[col].clip(lower, upper)

            elif action == 'TRANSFORMACAO_LOG':
                df[col] = np.log(df[col] + 1)

            elif action == 'TRANSFORMACAO_REFLEXAO_LOG':
                max_val = df[col].max()
                df[col] = np.log(max_val + 1 - df[col] + 1e-6)

            elif action == 'REMOCAO_OUTLIERS':
                lower = params['q1'] - 1.5 * params['iqr']
                upper = params['q3'] + 1.5 * params['iqr']
                df = df[(df[col] >= lower) & (df[col] <= upper)]

            elif action == 'NENHUMA':
                continue

            else:
                raise NotImplementedError(f"Ação '{action}' não reconhecida no tratamento de outliers.")

        return df

