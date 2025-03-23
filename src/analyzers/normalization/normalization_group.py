import pandas as pd
from src.analyzers.normalization.normalization_analyzer import NormalizationAnalyzer
from src.analyzers.normalization.normalization_plan import NormalizationPlan
from src.analyzers.normalization.normalization_executor import NormalizationExecutor
from src.analyzers.normalization.normalization_report import NormalizationReport

class NormalizationGroup:
    def __init__(self):
        self.analyzer = NormalizationAnalyzer()
        self.planner = NormalizationPlan()
        self.executor = NormalizationExecutor()
        self.reporter = NormalizationReport()

    def run(self, df: pd.DataFrame) -> dict:
        # 1. Análise da necessidade de padronização
        analysis = self.analyzer.analyze(df)

        # 2. Geração do plano de padronização
        plan = self.planner.generate(analysis)

        # 3. Execução do plano (aplicação dos scalers)
        transformed_data = self.executor.execute(df, plan)

        # 4. Geração do relatório
        report = self.reporter.generate(plan)

        return {
            'data': transformed_data,
            'analysis': analysis,
            'plan': plan,
            'report': report
        }