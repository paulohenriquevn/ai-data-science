class FeatureImportancePlan:
    def __init__(self, exclude_columns=None):
        self.exclude_columns = exclude_columns or ["id"]

    def generate(self, analysis: list) -> list:
        plan = []

        for item in analysis:
            col = item.get("column")
            if col in self.exclude_columns:
                continue

            plan.append({
                'column': col,
                'problem': item.get("problem", "low_feature_importance"),
                'suggestion': item.get("suggestion"),
                'all_actions': item.get("actions", []),
                'parameters': item.get("statistics", {})
            })

        return plan

# Exemplo de uso:
# planner = FeatureImportancePlan()
# feature_importance_plan = planner.generate(importance_result)
# feature_importance_plan[:5]
