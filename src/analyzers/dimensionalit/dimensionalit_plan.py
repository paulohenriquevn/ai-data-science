class DimensionalityPlan:
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
                'problem': item.get("problem", "dimensionality_reduction"),
                'suggestion': item.get("suggestion"),
                'all_actions': item.get("actions", []),
                'parameters': item.get("statistics", {})
            })

        return plan

