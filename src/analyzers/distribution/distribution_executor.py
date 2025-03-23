import numpy as np
import pandas as pd
from src.analyzers.base.analysis_base import ExecutionStep

class DistributionExecutor(ExecutionStep):
    def __init__(self, inplace: bool = True):
        self.inplace = inplace
        self.epsilon = 1e-6  # Para evitar divisões por zero e log(0)

    def execute(self, df: pd.DataFrame, plan: list) -> pd.DataFrame:
        df = df.copy()

        for step in plan:
            col = step['column']
            action = step['suggestion']

            if col not in df.columns:
                continue

            try:
                if action == 'TRANSFORMACAO_LOG':
                    df[col] = np.log(df[col] + 1)

                elif action == 'TRANSFORMACAO_RAIZ_QUADRADA':
                    df[col] = np.sqrt(df[col].clip(lower=0))

                elif action == 'TRANSFORMACAO_BOX_COX':
                    # Box-Cox só aceita valores positivos
                    mask = df[col] > 0
                    transformed, _ = stats.boxcox(df[col][mask])
                    df.loc[mask, col] = transformed

                elif action == 'TRANSFORMACAO_REFLECAO_LOG':
                    max_val = df[col].max()
                    df[col] = np.log(max_val + 1 - df[col] + self.epsilon)

                elif action == 'TRANSFORMACAO_QUADRATICA':
                    df[col] = df[col] ** 2

                elif action == 'TRANSFORMACAO_INVERSA':
                    df[col] = 1 / (df[col] + self.epsilon)

                elif action == 'CENTRALIZACAO_PADRONIZACAO':
                    df[col] = (df[col] - df[col].mean()) / df[col].std()

                elif action == 'WINSORIZATION':
                    q1 = df[col].quantile(0.05)
                    q9 = df[col].quantile(0.95)
                    df[col] = df[col].clip(q1, q9)

                elif action == 'CLIPPING':
                    min_val = df[col].quantile(0.01)
                    max_val = df[col].quantile(0.99)
                    df[col] = df[col].clip(min_val, max_val)

                elif action in ['ALGORITMO_ROBUSTO', 'NENHUMA']:
                    continue

                else:
                    raise NotImplementedError(f"Ação '{action}' não reconhecida.")

            except Exception as e:
                print(f"[Erro ao aplicar {action} na coluna {col}]: {str(e)}")

        return df
