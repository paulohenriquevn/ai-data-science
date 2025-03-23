import pandas as pd
from src.analyzers.low_variability.low_variability_analyzer import LowVariabilityAnalyzer
from src.analyzers.low_variability.low_variability_plan import LowVariabilityPlan
from src.analyzers.low_variability.low_variability_executor import LowVariabilityExecutor
from src.analyzers.low_variability.low_variability_report import LowVariabilityReport


class LowVariabilityGroup:
    def __init__(self):
        self.analyzer = LowVariabilityAnalyzer()
        self.planner = LowVariabilityPlan()
        self.executor = LowVariabilityExecutor()
        self.reporter = LowVariabilityReport()

    def run(self, df: pd.DataFrame) -> dict:
        # 1. Análise de baixa variabilidade
        analysis = self.analyzer.analyze(df)

        # 2. Geração do plano
        plan = self.planner.generate(analysis)

        # 3. Execução do plano
        transformed_data = self.executor.execute(df, plan)

        # 4. Relatório
        report = self.reporter.generate(plan)

        return {
            'data': transformed_data,
            'analysis': analysis,
            'plan': plan,
            'report': report
        }
