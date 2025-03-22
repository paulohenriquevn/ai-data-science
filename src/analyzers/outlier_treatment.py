from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd


class OutlierTreatmentTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, treatment_plan=None, log_offset=1e-6):
        """
        Transformador para tratamento de outliers com suporte a:
        - winsorização
        - remoção
        - transformação logarítmica

        Args:
            treatment_plan (dict): {coluna: 'log' | 'winsorize' | 'remove'}
            log_offset (float): valor somado antes do log para evitar log(0)
        """
        self.treatment_plan = treatment_plan or {}
        self.log_offset = log_offset
        self.fitted_ = False
        self.bounds_ = {}

    def fit(self, X, y=None):
        X = X.copy()

        for col in X.select_dtypes(include=[np.number]).columns:
            if col not in self.treatment_plan:
                continue

            strategy = self.treatment_plan[col]

            if strategy in ['winsorize', 'remove']:
                q1 = X[col].quantile(0.25)
                q3 = X[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                self.bounds_[col] = (lower, upper)

        self.fitted_ = True
        return self

    def transform(self, X):
        if not self.fitted_:
            raise ValueError("É necessário executar .fit() antes de .transform()")

        X = X.copy()

        for col, strategy in self.treatment_plan.items():
            if col not in X.columns:
                continue

            if strategy == 'log':
                # Garante que todos os valores estejam acima de -offset para evitar log negativo
                if (X[col] + self.log_offset <= 0).any():
                    raise ValueError(f"Coluna {col} possui valores negativos que impedem a aplicação de log.")
                X[col] = np.log(X[col] + self.log_offset)

            elif strategy == 'winsorize':
                lower, upper = self.bounds_.get(col, (None, None))
                X[col] = X[col].clip(lower, upper)

            elif strategy == 'remove':
                lower, upper = self.bounds_.get(col, (None, None))
                X = X[(X[col] >= lower) & (X[col] <= upper)]

        return X