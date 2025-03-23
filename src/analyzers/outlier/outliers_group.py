import pandas as pd
from src.analyzers.outlier.outliers_analyzer import OutlierAnalyzer
from src.analyzers.outlier.outliers_plan import OutlierPlan
from src.analyzers.outlier.outliers_executor import OutlierExecutor
from src.analyzers.outlier.outliers_report import OutlierReport

class OutlierGroup:
    def __init__(self):
        self.analyzer = OutlierAnalyzer()
        self.planner = OutlierPlan()
        self.executor = OutlierExecutor()
        self.reporter = OutlierReport()

    def run(self, df: pd.DataFrame) -> dict:
        # 1. Análise de outliers
        analysis = self.analyzer.analyze(df)

        # 2. Geração do plano
        plan = self.planner.generate(analysis)

        # 3. Execução do plano
        transformed_data = self.executor.execute(df, plan)

        # 4. Geração do relatório
        report = self.reporter.generate(plan)

        return {
            'data': transformed_data,
            'analysis': analysis,
            'plan': plan,
            'report': report
        }


