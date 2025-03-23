import pandas as pd
from src.analyzers.base.analysis_base import PlanStep

class DistributionPlan(PlanStep):
    def generate(self, analysis_output: list) -> list:
        plan = []

        for item in analysis_output:
            plan.append({
                'column': item['column'],
                'problem': item['problem'],
                'suggestion': item['suggestion'],
                'all_actions': item['actions'],
                'parameters': item['statistics']
            })

        return plan
