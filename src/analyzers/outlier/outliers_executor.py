from src.analyzers.base.analysis_base import ExecutionStep

class OutliersExecutor(ExecutionStep):
    def execute(self, X, y=None):
        X = X.copy()
        return X