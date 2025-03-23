import pandas as pd
from src.analyzers.residual_collinearity.residual_collinearity_analyzer import ResidualCollinearityAnalyzer
from src.analyzers.residual_collinearity.residual_collinearity_plan import ResidualCollinearityPlan
from src.analyzers.residual_collinearity.residual_collinearity_executor import ResidualCollinearityExecutor
from src.analyzers.residual_collinearity.residual_collinearity_report import ResidualCollinearityReport

class ResidualCollinearityGroup:
    def __init__(self, target_column: str):
        self.analyzer = ResidualCollinearityAnalyzer(target_column=target_column)
        self.planner = ResidualCollinearityPlan()
        self.executor = ResidualCollinearityExecutor()
        self.reporter = ResidualCollinearityReport()

    def run(self, df: pd.DataFrame) -> dict:
        # 1. Análise de colinearidade residual
        analysis = self.analyzer.analyze(df)

        # 2. Plano de ação
        plan = self.planner.generate(analysis)

        # 3. Execução
        transformed_data = self.executor.execute(df, plan)

        # 4. Relatório
        report = self.reporter.generate(plan)

        return {
            "data": transformed_data,
            "analysis": analysis,
            "plan": plan,
            "report": report
        }
