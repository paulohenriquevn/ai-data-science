import pandas as pd
from src.analyzers.rare_category.rare_category_analyzer import RareCategoryAnalyzer
from src.analyzers.rare_category.rare_category_plan import RareCategoryPlan
from src.analyzers.rare_category.rare_category_executor import RareCategoryExecutor
from src.analyzers.rare_category.rare_category_report import RareCategoryReport


class RareCategoryGroup:
    def __init__(self):
        self.analyzer = RareCategoryAnalyzer()
        self.planner = RareCategoryPlan()
        self.executor = RareCategoryExecutor()
        self.reporter = RareCategoryReport()

    def run(self, df: pd.DataFrame) -> dict:
        # 1. Análise de categorias raras
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
