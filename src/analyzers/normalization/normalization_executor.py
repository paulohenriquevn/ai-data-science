from sklearn.preprocessing import StandardScaler, RobustScaler
from src.analyzers.base.analysis_base import ExecutionStep

class NormalizationExecutor(ExecutionStep):
    def __init__(self, normalization_plan):
        """
        Args:
            normalization_plan: dict do tipo {coluna: "StandardScaler" | "RobustScaler"}
        """
        self.normalization_plan = normalization_plan
        self.scalers = {}
        self.columns_to_scale = list(normalization_plan.keys())

    def fit(self, df):
        df = df.copy()
        for col in self.columns_to_scale:
            method = self.normalization_plan.get(col)
            if method == "RobustScaler":
                scaler = RobustScaler()
            else:
                scaler = StandardScaler()
            self.scalers[col] = scaler.fit(df[[col]])
        return self

    def transform(self, df):
        df = df.copy()
        for col in self.columns_to_scale:
            scaler = self.scalers.get(col)
            if scaler:
                df[col + "_scaled"] = scaler.transform(df[[col]])
        return df
