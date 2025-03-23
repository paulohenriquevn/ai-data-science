import pandas as pd
from src.analyzers.base.analysis_base import GroupStep
from src.analyzers.collect.collect_analyzer import CollectAnalyzer
from src.analyzers.collect.collect_plan import CollectPlan
from src.analyzers.collect.collect_executor import CollectExecutor
from src.analyzers.collect.collect_report import CollectReport

class CollectGroup(GroupStep):
    def __init__(self):
        self.analyzer = CollectAnalyzer()
        self.planner = CollectPlan()
        self.executor = CollectExecutor()
        self.reporter = CollectReport()

    def run(self, data: pd.DataFrame) -> dict:
        # 1. Análise
        analysis = self.analyzer.analyze(data)

        # 2. Geração do plano
        plan = self.planner.generate(analysis)

        # 3. Execução das ações
        transformed_data = self.executor.execute(data, plan)

        # 4. Geração do relatório
        report = self.reporter.generate(plan)

        return {
            'data': transformed_data,
            'analysis': analysis,
            'plan': plan,
            'report': report
        }