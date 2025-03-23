import pandas as pd
from src.analyzers.feature_Importance.feature_Importance_analyzer import FeatureImportanceAnalyzer
from src.analyzers.feature_Importance.feature_Importance_plan import FeatureImportancePlan
from src.analyzers.feature_Importance.feature_Importance_executor import FeatureImportanceExecutor
from src.analyzers.feature_Importance.feature_Importance_report import FeatureImportanceReport


class FeatureImportanceGroup:
    def __init__(self, target_column: str):
        self.analyzer = FeatureImportanceAnalyzer(target_column=target_column)
        self.planner = FeatureImportancePlan()
        self.executor = FeatureImportanceExecutor()
        self.reporter = FeatureImportanceReport()

    def run(self, df: pd.DataFrame) -> dict:
        # 1. Análise de importância de variáveis
        analysis = self.analyzer.analyze(df)

        # 2. Geração do plano de ação
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
