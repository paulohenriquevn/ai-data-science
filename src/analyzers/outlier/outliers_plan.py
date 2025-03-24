class OutlierPlan:
    def __init__(self, exclude_columns=None, include_columns=None):
        self.exclude_columns = exclude_columns or ["id"]
        self.include_columns = include_columns  # nova opção para controle manual

    def generate(self, analysis: list) -> list:
        plan = []

        for item in analysis:
            col = item.get("column")

            if self.include_columns is not None and col not in self.include_columns:
                continue  # ignora colunas não selecionadas

            if col in self.exclude_columns:
                continue  # ignora colunas excluídas

            plan.append({
                'column': col,
                'problem': item.get("problem", "outlier_detection"),
                'suggestion': item.get("suggestion"),
                'all_actions': item.get("actions", []),
                'parameters': item.get("statistics", {})
            })

        return plan