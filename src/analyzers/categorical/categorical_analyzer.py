import pandas as pd
from src.analyzers.base.analysis_base import AnalysisStep


class CategoricalAnalyzer(AnalysisStep):
    def __init__(self, cardinality_threshold: int = 10):
        self.cardinality_threshold = cardinality_threshold

    def analyze(self, df: pd.DataFrame) -> list:
        results = []
        object_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        for col in object_cols:
            series = df[col]
            nunique = series.nunique(dropna=True)

            if nunique <= 1:
                suggestion = "NENHUMA"
                actions = ["NENHUMA"]
                description = f"Coluna '{col}' possui apenas um valor distinto."
            elif nunique <= self.cardinality_threshold:
                suggestion = "ONE_HOT"
                actions = ["ONE_HOT", "ORDINAL", "NUMERIC_CAST"]
                description = f"Coluna '{col}' com baixa cardinalidade ({nunique} categorias)."
            elif nunique <= 50:
                suggestion = "ORDINAL"
                actions = ["ORDINAL", "NUMERIC_CAST"]
                description = f"Coluna '{col}' com cardinalidade moderada ({nunique} categorias)."
            else:
                suggestion = "NUMERIC_CAST"
                actions = ["NUMERIC_CAST", "ORDINAL"]
                description = f"Coluna '{col}' com alta cardinalidade ({nunique} categorias)."

            results.append({
                'column': col,
                'problem': 'categorical_encoding',
                'problem_description': description,
                'suggestion': suggestion,
                'actions': actions,
                'statistics': {
                    'num_categories': nunique,
                    'dtype': str(series.dtype)
                }
            })

        return results
