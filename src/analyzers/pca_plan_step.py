import numpy as np
from sklearn.decomposition import PCA


class PCAPlanStep:
    def __init__(self, df, significance_report=None, correlation_report=None, exclude_columns=None, variance_threshold=0.90):
        self.df = df
        self.significance_report = significance_report or []
        self.correlation_report = correlation_report or []
        self.exclude_columns = exclude_columns or ["id", "y"]
        self.variance_threshold = variance_threshold

    def generate_plan(self):
        plan = {
            "pca_candidate_columns": [],
            "pca_variance_curve": [],
            "recommended_n_components": None
        }

        # Selecionar variáveis significativas e correlacionadas para PCA
        sig_vars = {r["column"] for r in self.significance_report if r.get("statistics", {}).get("p_value", 1) < 0.05}
        corr_vars = {r["column"] for r in self.correlation_report if r.get("statistics", {}).get("correlation_strength") in ["MODERADA", "FORTE"]}
        candidate_vars = list(sig_vars & corr_vars)

        scaled_cols = [col for col in self.df.columns if col.endswith("_scaled") and col not in self.exclude_columns]
        plan["pca_candidate_columns"] = [col for col in scaled_cols if any(core in col for core in candidate_vars)]

        if len(plan["pca_candidate_columns"]) > 0:
            X_pca = self.df[plan["pca_candidate_columns"]].dropna()
            pca = PCA()
            pca.fit(X_pca)
            cumulative_variance = pca.explained_variance_ratio_.cumsum()
            plan["pca_variance_curve"] = cumulative_variance.tolist()
            plan["recommended_n_components"] = int((cumulative_variance >= self.variance_threshold).argmax() + 1)

        return plan
