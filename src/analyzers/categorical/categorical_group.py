import pandas as pd
from src.analyzers.categorical.categorical_analyzer import CategoricalAnalyzer
from src.analyzers.categorical.categorical_plan import CategoricalPlan
from src.analyzers.categorical.categorical_executor import CategoricalExecutor
from src.analyzers.categorical.categorical_report import CategoricalReport

class CategoricalGroup:
    def __init__(self):
        self.analyzer = CategoricalAnalyzer()
        self.planner = CategoricalPlan()
        self.executor = CategoricalExecutor()
        self.reporter = CategoricalReport()

    def run(self, df: pd.DataFrame) -> dict:
        # 1. Análise das variáveis categóricas
        analysis = self.analyzer.analyze(df)

        # 2. Geração do plano de codificação
        plan = self.planner.generate(analysis)

        # 3. Execução do plano (codificação aplicada)
        transformed_data = self.executor.execute(df, plan)

        # 4. Geração do relatório
        report = self.reporter.generate(plan)

        return {
            'data': transformed_data,
            'analysis': analysis,
            'plan': plan,
            'report': report
        }

