import pandas as pd
from src.analyzers.feature_interaction.feature_interaction_analyzer import FeatureInteractionAnalyzer
from src.analyzers.feature_interaction.feature_interaction_plan import FeatureInteractionPlan
from src.analyzers.feature_interaction.feature_interaction_executor import FeatureInteractionExecutor
from src.analyzers.feature_interaction.feature_interaction_report import FeatureInteractionReport


class FeatureInteractionGroup:
    def __init__(self, target_column: str):
        self.analyzer = FeatureInteractionAnalyzer(target_column=target_column)
        self.planner = FeatureInteractionPlan()
        self.executor = FeatureInteractionExecutor()
        self.reporter = FeatureInteractionReport()

    def run(self, df: pd.DataFrame) -> dict:
        # 1. Análise de interações potenciais
        analysis = self.analyzer.analyze(df)

        # 2. Geração do plano de criação de features
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
