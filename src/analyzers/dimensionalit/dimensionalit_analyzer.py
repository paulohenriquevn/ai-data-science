import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler


class DimensionalityAnalyzer:
    def __init__(self, vif_threshold: float = 10.0, variance_threshold: float = 0.01, max_features: int = 20):
        self.vif_threshold = vif_threshold
        self.variance_threshold = variance_threshold
        self.max_features = max_features

    def analyze(self, df: pd.DataFrame) -> list:
        results = []
        numeric_df = df.select_dtypes(include="number").dropna(axis=1, how="any")

        # Remover colunas com valores não finitos
        numeric_df = numeric_df.loc[:, numeric_df.apply(lambda x: np.isfinite(x).all())]

        if numeric_df.shape[1] < 2:
            return []

        # Padronizar para análise
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(numeric_df), columns=numeric_df.columns)

        # Etapa 1: Seleção com VarianceThreshold
        selector = VarianceThreshold(threshold=self.variance_threshold)
        try:
            reduced_df = pd.DataFrame(
                selector.fit_transform(X_scaled),
                columns=X_scaled.columns[selector.get_support()]
            )
        except Exception:
            return []

        # Etapa 2: Redução por alta correlação
        corr_matrix = reduced_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        reduced_df.drop(columns=to_drop, errors="ignore", inplace=True)

        # Etapa 3: Limitar número de colunas analisadas
        if reduced_df.shape[1] > self.max_features:
            reduced_df = reduced_df.iloc[:, :self.max_features]

        # Etapa 4: Amostragem de linhas
        sampled = reduced_df.sample(min(1000, len(reduced_df)), random_state=42)

        # Etapa 5: Calcular VIF
        vif_scores = pd.Series(
            [variance_inflation_factor(sampled.values, i) for i in range(sampled.shape[1])],
            index=sampled.columns
        )

        for col in sampled.columns:
            vif = vif_scores[col]
            var = sampled[col].var()
            std = sampled[col].std()

            actions = []
            if vif > self.vif_threshold:
                actions += ['PCA', 'VIF_REDUCTION']
            if var < self.variance_threshold:
                actions += ['VARIANCE_THRESHOLD']
            if not actions:
                continue

            suggestion = actions[0]
            results.append({
                'column': col,
                'problem': 'dimensionality_reduction',
                'problem_description': f'Coluna apresenta {"alta colinearidade (VIF = " + str(round(vif, 2)) + ")" if vif > self.vif_threshold else ""}{" e " if vif > self.vif_threshold and var < self.variance_threshold else ""}{"variância muito baixa" if var < self.variance_threshold else ""}.',
                'suggestion': suggestion,
                'actions': list(set(actions)),
                'statistics': {
                    'vif': float(vif),
                    'std': float(std),
                    'variance': float(var)
                }
            })

        return results