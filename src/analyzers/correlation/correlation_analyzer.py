from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import numpy as np


class CorrelationAnalyzer:
    def __init__(self, target_column=None, correlation_threshold=0.9, vif_threshold=10, max_vif_features=20):
        self.target_column = target_column
        self.correlation_threshold = correlation_threshold
        self.vif_threshold = vif_threshold
        self.max_vif_features = max_vif_features

    def analyze(self, df: pd.DataFrame) -> list:
        results = []
        numeric_df = df.select_dtypes(include=['number'])

        if self.target_column and self.target_column in numeric_df.columns:
            results += self._analyze_target_correlation(numeric_df)

        results += self._analyze_pairwise_correlation(numeric_df)
        return results

    def _analyze_target_correlation(self, df: pd.DataFrame) -> list:
        target = self.target_column
        results = []
        correlations = df.corr()[target].drop(target)

        for col, corr_value in correlations.items():
            nivel, sentido = self._classificar_correlacao(corr_value)
            actions = CorrelationScenario.recommendations(nivel)
            suggestion = actions[0] if actions else 'NENHUMA'

            results.append({
                'column': col,
                'problem': 'correlation_pattern',
                'problem_description': f'Correlação com a variável alvo ({target}) é {sentido} e {nivel} ({corr_value:.2f})',
                'suggestion': suggestion,
                'actions': actions,
                'statistics': {
                    'correlation': corr_value,
                    'sentido': sentido,
                    'nivel_relevancia': nivel,
                    'target_column': target
                }
            })

        return results

    def _analyze_pairwise_correlation(self, df: pd.DataFrame) -> list:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from sklearn.preprocessing import StandardScaler

        df_clean = df.dropna(axis=1, how='any')
        df_clean = df_clean.loc[:, df_clean.apply(lambda col: np.isfinite(col).all())]

        # Etapa 1: Selecionar colunas com variância suficiente
        selector = VarianceThreshold(threshold=0.01)
        try:
            reduced_df = pd.DataFrame(selector.fit_transform(df_clean), columns=df_clean.columns[selector.get_support()])
        except Exception:
            return []

        # Etapa 2: Selecionar colunas com baixa correlação entre si (redução adicional)
        corr_matrix = reduced_df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
        reduced_df = reduced_df.drop(columns=to_drop, errors="ignore")

        # Etapa 3: Amostrar linhas e limitar colunas
        if reduced_df.shape[1] > self.max_vif_features:
            reduced_df = reduced_df.iloc[:, :self.max_vif_features]

        sampled = reduced_df.sample(min(1000, len(reduced_df)), random_state=42)
        scaler = StandardScaler()
        scaled = pd.DataFrame(scaler.fit_transform(sampled), columns=sampled.columns)

        # Etapa 4: Calcular VIF
        vif_data = pd.DataFrame()
        vif_data["feature"] = scaled.columns
        vif_data["VIF"] = [variance_inflation_factor(scaled.values, i) for i in range(scaled.shape[1])]

        results = []
        for _, row in vif_data.iterrows():
            if row["VIF"] >= self.vif_threshold:
                feature = row["feature"]
                results.append({
                    'column': feature,
                    'problem': 'correlation_pattern',
                    'problem_description': f'Variável apresenta multicolinearidade (VIF = {row["VIF"]:.2f})',
                    'suggestion': 'REGULARIZATION',
                    'actions': ['REGULARIZATION', 'STEPWISE_SELECTION', 'PCA'],
                    'statistics': {
                        'vif': row["VIF"]
                    }
                })

        return results

    def _classificar_correlacao(self, corr: float) -> tuple:
        sentido = "positivo" if corr > 0 else "negativo"
        abs_corr = abs(corr)

        if abs_corr >= 0.9:
            return "muito alta", sentido
        elif abs_corr >= 0.7:
            return "alta", sentido
        elif abs_corr >= 0.5:
            return "moderada", sentido
        elif abs_corr >= 0.3:
            return "baixa", sentido
        else:
            return "muito baixa", sentido


class CorrelationScenario:
    @staticmethod
    def recommendations(nivel: str) -> list:
        mapping = {
            'muito alta': ['EXCLUIR_COLUNA', 'PCA'],
            'alta': ['REGULARIZATION', 'PCA', 'STEPWISE_SELECTION'],
            'moderada': ['STEPWISE_SELECTION', 'REGULARIZATION'],
            'baixa': ['NENHUMA'],
            'muito baixa': ['NENHUMA']
        }
        return mapping.get(nivel.lower(), ['NENHUMA'])


