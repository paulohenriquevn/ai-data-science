from src.analyzers.base.analysis_base import ReportStep
import pandas as pd

class CollectReport(ReportStep):
    def generate(self, plan: list) -> list:
        report = []

        for step in plan:
            column = step['column']
            problem = step['problem']
            action = step['suggestion']
            stats = step.get('parameters', {})

            description = self._describe_action(problem, action, column)

            report.append({
                'column': column,
                'problem': problem,
                'action_taken': action,
                'description': description,
                'statistics': stats
            })

        return report

    def _describe_action(self, problem, action, column):
        """
        Gera uma descrição legível para a ação aplicada.
        """
        desc_map = {
            'FILL_NA': f'Foram preenchidos valores ausentes com 0 na coluna "{column}".',
            'DROP_NA': f'Foram removidas linhas com valores ausentes na coluna "{column}".',
            'REPLACE_PLACEHOLDERS_WITH_NAN': f'Valores reservados foram substituídos por NaN na coluna "{column}".',
            'REPLACE_NUMERIC_PLACEHOLDERS_WITH_NAN': f'Placeholders numéricos suspeitos foram substituídos por NaN na coluna "{column}".',
            'CONVERT_TO_INTEGER': f'A coluna "{column}" foi convertida para tipo inteiro.',
            'CONVERT_TO_FLOAT': f'A coluna "{column}" foi convertida para tipo float.',
            'CONVERT_TO_BOOLEAN': f'A coluna "{column}" foi convertida para booleano.',
            'CONVERT_TO_DATETIME': f'A coluna "{column}" foi convertida para data/hora.',
            'EVALUATE_ENCODING': f'A coluna "{column}" foi marcada para avaliação de encoding categórico.'
        }

        return desc_map.get(action, f'Ação "{action}" aplicada na coluna "{column}".')