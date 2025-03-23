class FeatureInteractionPlan:
    def __init__(self):
        pass

    def generate(self, analysis: list) -> list:
        plan = []

        for item in analysis:
            col1, col2 = item.get("column", [None, None])
            if not col1 or not col2:
                continue

            plan.append({
                'column': [col1, col2],
                'problem': item.get("problem", "potential_interaction"),
                'suggestion': item.get("suggestion"),
                'all_actions': item.get("actions", []),
                'parameters': item.get("statistics", {})
            })

        return plan
