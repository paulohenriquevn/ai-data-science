class BalancePlanStep:
    def __init__(self, balance_report: dict, strategy_priority: list = None):
        self.balance_report = balance_report or {}
        self.strategy_priority = strategy_priority or [
            "SMOTE", "OVERSAMPLING", "UNDERSAMPLING", "AUMENTAR_DADOS"
        ]

    def generate_plan(self):
        plan = {
            "apply_balance": False,
            "strategy": None,
            "justification": ""
        }

        problem = self.balance_report.get("problem")
        actions = self.balance_report.get("actions", [])
        ratio = self.balance_report.get("imbalance_ratio")

        if problem == "BALANCEADO":
            plan["justification"] = "Distribuição das classes está balanceada. Nenhuma ação necessária."
            return plan

        plan["apply_balance"] = True
        plan["justification"] = f"Desequilíbrio detectado (ratio = {ratio})."

        for strategy in self.strategy_priority:
            if strategy in actions:
                plan["strategy"] = strategy
                break

        return plan


# Exemplo de uso:
# balance_report = BalanceAnalyzer().analyze(df)
# plan = BalancePlanStep(balance_report).generate_plan()
# print(plan)
