import numpy as np
from sklearn.decomposition import PCA

class NormalizationPlanStep:
    def __init__(self, distribution_report=None, outlier_report=None, exclude_columns=None, skew_threshold=1.0):
        self.distribution_report = distribution_report or []
        self.outlier_report = outlier_report or []
        self.exclude_columns = exclude_columns or ["id", "y"]
        self.skew_threshold = skew_threshold

    def generate_plan(self):
        plan = {}
        for item in self.distribution_report:
            col = item.get("column")
            if col in self.exclude_columns:
                continue
            skew = item.get("statistics", {}).get("skewness", 0)
            if abs(skew) > self.skew_threshold:
                plan[col] = "RobustScaler"
            else:
                plan[col] = "StandardScaler"

        # Ajuste baseado em outliers (reforçar RobustScaler se necessário)
        for item in self.outlier_report:
            col = item.get("column")
            if col in self.exclude_columns:
                continue
            ratio = item.get("statistics", {}).get("outlier_ratio", 0)
            if ratio > 0.1:
                plan[col] = "RobustScaler"

        return plan