import pandas as pd
from src.analyzers.distribution.distribution_analyzer import DistributionAnalyzer
from src.analyzers.distribution.distribution_plan import DistributionPlan
from src.analyzers.distribution.distribution_executor import DistributionExecutor
from src.analyzers.distribution.distribution_report import DistributionReport
from src.analyzers.base.analysis_base import GroupStep


class DistributionGroup(GroupStep):
    def __init__(self):
        self.analyzer = DistributionAnalyzer()
        self.planner = DistributionPlan()
        self.executor = DistributionExecutor()
        self.reporter = DistributionReport()

    def run(self, data: pd.DataFrame) -> dict:
        # 1. Análise das distribuições
        analysis = self.analyzer.analyze(data)

        # 2. Geração do plano
        plan = self.planner.generate(analysis)

        # 3. Execução das transformações
        transformed_data = self.executor.execute(data, plan)

        # 4. Geração do relatório
        report = self.reporter.generate(plan)

        return {
            'data': transformed_data,
            'analysis': analysis,
            'plan': plan,
            'report': report
        }