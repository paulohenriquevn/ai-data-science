from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

class ResidualCollinearityAnalyzer:
    def __init__(self, target_column: str, vif_threshold: float = 10.0, corr_threshold: float = 0.95):
        self.target_column = target_column
        self.vif_threshold = vif_threshold
        self.corr_threshold = corr_threshold

    def analyze(self, df: pd.DataFrame) -> list:
        if self.target_column not in df.columns:
            raise ValueError(f"A coluna alvo '{self.target_column}' não foi encontrada no DataFrame.")

        results = []
        numeric_df = df.select_dtypes(include="number").drop(columns=[self.target_column], errors="ignore")
        numeric_df = numeric_df.dropna(axis=1, how="any")  # excluir colunas com NaNs

        if numeric_df.shape[1] < 2:
            return results

        # Calcular matriz de correlação
        corr_matrix = numeric_df.corr().abs()
        y_corr = df[numeric_df.columns].corrwith(df[self.target_column]).abs()

        # Calcular VIF
        vif_data = pd.DataFrame()
        vif_data["feature"] = numeric_df.columns
        vif_data["vif"] = [variance_inflation_factor(numeric_df.values, i) for i in range(numeric_df.shape[1])]

        for _, row in vif_data.iterrows():
            col = row["feature"]
            vif = row["vif"]
            max_corr = corr_matrix[col].drop(col).max()
            target_corr = y_corr[col]

            actions = []
            if vif > self.vif_threshold:
                actions.append("REMOVER_VARIAVEL")
            if max_corr > self.corr_threshold:
                actions.append("REMOVER_REDUNDANTE")
            if target_corr < 0.01 and (vif > self.vif_threshold or max_corr > self.corr_threshold):
                actions.append("EXCLUIR_POR_RELEVANCIA_BAIXA")

            if not actions:
                continue

            suggestion = actions[0]

            results.append({
                'column': col,
                'problem': 'residual_collinearity',
                'problem_description': f'VIF = {vif:.2f}, correlação máx. com outra variável = {max_corr:.2f}, correlação com y = {target_corr:.2f}',
                'suggestion': suggestion,
                'actions': list(set(actions)),
                'statistics': {
                    'vif': float(vif),
                    'max_corr_with_other': float(max_corr),
                    'correlation_with_target': float(target_corr)
                }
            })

        return results

