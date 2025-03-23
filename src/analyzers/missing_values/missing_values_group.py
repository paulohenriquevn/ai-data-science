import pandas as pd
from src.analyzers.base.analysis_base import GroupStep
from src.analyzers.missing_values.missing_values_analyzer import MissingValuesAnalyzer
from src.analyzers.missing_values.missing_values_plan import MissingValuesPlan
from src.analyzers.missing_values.missing_values_executor import MissingValuesExecutor
from src.analyzers.missing_values.missing_values_report import MissingValuesReport



class MissingGroup(GroupStep):
    def __init__(self):
        self.analyzer = MissingValuesAnalyzer()
        self.planner = MissingValuesPlan()
        self.executor = MissingValuesExecutor()
        self.reporter = MissingValuesReport()

    def run(self, data: pd.DataFrame) -> dict:
        # Etapa 1: Análise
        analysis = self.analyzer.analyze(data)

        # Etapa 2: Plano
        plan = self.planner.generate(analysis)

        # Etapa 3: Execução
        transformed_data = self.executor.execute(data, plan)

        # Etapa 4: Relatório
        report = self.reporter.generate(plan)

        return {
            'data': transformed_data,
            'analysis': analysis,
            'plan': plan,
            'report': report
        }

