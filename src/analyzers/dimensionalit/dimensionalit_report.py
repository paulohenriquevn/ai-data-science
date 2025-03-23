class DimensionalityReport:
    def generate(self, plan: list) -> list:
        report = []

        for step in plan:
            column = step['column']
            action = step['suggestion']
            stats = step.get('parameters', {})
            description = self._describe_action(action, column)

            report.append({
                'column': column,
                'problem': 'dimensionality_reduction',
                'action_taken': action,
                'description': description,
                'statistics': stats
            })

        return report

    def _describe_action(self, action: str, column: str) -> str:
        desc_map = {
            'PCA': f'Coluna "{column}" foi substituída por componente principal via PCA.',
            'VIF_REDUCTION': f'Coluna "{column}" marcada para redução de VIF.',
            'VARIANCE_THRESHOLD': f'Coluna "{column}" removida por baixa variância.',
            'NENHUMA': f'Nenhuma transformação aplicada à coluna "{column}".'
        }
        return desc_map.get(action, f'Ação "{action}" aplicada à coluna "{column}".')

