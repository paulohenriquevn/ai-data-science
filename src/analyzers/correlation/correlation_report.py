class CorrelationReport:
    def generate(self, plan: list) -> list:
        report = []

        for step in plan:
            column = step['column']
            action = step['suggestion']
            stats = step.get('parameters', {})
            description = self._describe_action(action, column)

            report.append({
                'column': column,
                'problem': 'correlation_pattern',
                'action_taken': action,
                'description': description,
                'statistics': stats
            })

        return report

    def _describe_action(self, action: str, column: str) -> str:
        desc_map = {
            'EXCLUIR_COLUNA': f'Coluna "{column}" removida por multicolinearidade.',
            'PCA': f'Coluna "{column}" marcada para agrupamento via PCA.',
            'REGULARIZATION': f'Coluna "{column}" mantida com recomendação de regularização.',
            'STEPWISE_SELECTION': f'Coluna "{column}" marcada para análise stepwise.',
            'NENHUMA': f'Nenhuma ação aplicada à coluna "{column}".'
        }
        return desc_map.get(action, f'Ação "{action}" aplicada à coluna "{column}".')


