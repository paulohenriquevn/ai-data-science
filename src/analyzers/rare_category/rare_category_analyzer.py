import pandas as pd


class RareCategoryAnalyzer:
    def __init__(self, rare_threshold: float = 0.01, rare_ratio_limit: float = 0.1, cardinality_limit: int = 15):
        self.rare_threshold = rare_threshold
        self.rare_ratio_limit = rare_ratio_limit
        self.cardinality_limit = cardinality_limit

    def analyze(self, df: pd.DataFrame) -> list:
        results = []
        cat_cols = df.select_dtypes(include=['object', 'category']).columns

        for col in cat_cols:
            series = df[col].dropna()
            value_counts = series.value_counts(normalize=True)
            rare_categories = value_counts[value_counts < self.rare_threshold]
            rare_ratio = len(rare_categories) / len(value_counts)

            actions = []
            if rare_ratio > self.rare_ratio_limit:
                actions.append("AGRUPAR_RARAS")
            if len(value_counts) > self.cardinality_limit:
                actions.append("REDUZIR_CARDINALIDADE")

            if not actions:
                continue

            suggestion = actions[0]
            description = f"Categoria mais comum representa {value_counts.iloc[0]:.0%}, e {len(rare_categories)} categorias têm frequência < {self.rare_threshold:.0%}."

            results.append({
                'column': col,
                'problem': 'rare_category_detection',
                'problem_description': description,
                'suggestion': suggestion,
                'actions': actions,
                'statistics': {
                    'total_categories': len(value_counts),
                    'rare_categories_count': len(rare_categories),
                    'rare_categories_ratio': rare_ratio
                }
            })

        return results
