import pandas as pd
from src.analyzers.correlation.correlation_analyzer import CorrelationAnalyzer
from src.analyzers.correlation.correlation_plan import CorrelationPlan
from src.analyzers.correlation.correlation_executor import CorrelationExecutor
from src.analyzers.correlation.correlation_report import CorrelationReport
from src.analyzers.base.analysis_base import GroupStep

class CorrelationGroup(GroupStep):
    def __init__(self, target_column=None):
        self.analyzer = CorrelationAnalyzer(target_column=target_column)
        self.planner = CorrelationPlan()
        self.executor = CorrelationExecutor()
        self.reporter = CorrelationReport()

    def run(self, df: pd.DataFrame) -> dict:
        # 1. Análise de correlação
        analysis = self.analyzer.analyze(df)

        # 2. Geração do plano
        plan = self.planner.generate(analysis)

        # 3. Execução das ações
        transformed_data = self.executor.execute(df, plan)

        # 4. Geração do relatório
        report = self.reporter.generate(plan)

        return {
            'data': transformed_data,
            'analysis': analysis,
            'plan': plan,
            'report': report
        }
