import pandas as pd

class CategoricalPlan:
    """
    Gera um plano de transformação para variáveis categóricas baseado na análise realizada.
    """

    def __init__(self, analysis_result: dict, max_unique_for_ordinal=20):
        self.analysis_result = analysis_result
        self.max_unique_for_ordinal = max_unique_for_ordinal

    def generate(self) -> dict:
        """
        Recebe o resultado do analisador de variáveis categóricas e gera um plano de transformação.
        """
        plan = {}

        for item in self.analysis_result:
            col = item["column"]
            dtype = item.get("dtype")
            unique_values = item.get("n_unique")

            if dtype == "object":
                plan[col] = "ONE_HOT"
            elif dtype == "category":
                plan[col] = "ONE_HOT"
            elif dtype in ["int64", "float64"] and unique_values is not None and unique_values <= self.max_unique_for_ordinal:
                plan[col] = "ORDINAL"

        return plan
