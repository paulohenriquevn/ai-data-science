import pandas as pd
from src.analyzers.dimensionalit.dimensionalit_analyzer import DimensionalityAnalyzer
from src.analyzers.dimensionalit.dimensionalit_plan import DimensionalityPlan
from src.analyzers.dimensionalit.dimensionalit_executor import DimensionalityExecutor
from src.analyzers.dimensionalit.dimensionalit_report import DimensionalityReport

class DimensionalityGroup:
    def __init__(self):
        self.analyzer = DimensionalityAnalyzer()
        self.planner = DimensionalityPlan()
        self.executor = DimensionalityExecutor()
        self.reporter = DimensionalityReport()

    def run(self, df: pd.DataFrame) -> dict:
        # 1. Análise
        analysis = self.analyzer.analyze(df)

        # 2. Plano de ação
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
