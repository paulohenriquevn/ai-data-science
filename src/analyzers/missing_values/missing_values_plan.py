from src.analyzers.base.analysis_base import PlanStep


class MissingValuesPlan(PlanStep):
    def __init__(self, missing_report, exclude_columns=None):
        self.missing_report = missing_report or []
        self.exclude_columns = exclude_columns or ["id"]

    def generate(self):
        plan = {}
        for item in self.missing_report:
            col = item.get("column")
            if col in self.exclude_columns:
                continue
            solution = item.get("solution")
            if solution:
                plan[col] = solution.name  # Armazena a solução como string para uso posterior
        return plan
