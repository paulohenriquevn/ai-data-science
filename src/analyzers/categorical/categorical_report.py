class CategoricalReport:
    def generate(self, plan: list) -> list:
        report = []

        for step in plan:
            column = step['column']
            action = step['suggestion']
            stats = step.get('parameters', {})
            description = self._describe_action(action, column)

            report.append({
                'column': column,
                'problem': 'categorical_encoding',
                'action_taken': action,
                'description': description,
                'statistics': stats
            })

        return report

    def _describe_action(self, action: str, column: str) -> str:
        desc_map = {
            'ONE_HOT': f'Aplicada codificação One-Hot na coluna "{column}".',
            'ORDINAL': f'Aplicada codificação Ordinal na coluna "{column}".',
            'NUMERIC_CAST': f'Coluna "{column}" convertida para valor numérico com fallback.',
            'NENHUMA': f'Nenhuma codificação aplicada à coluna "{column}".'
        }
        return desc_map.get(action, f'Ação "{action}" aplicada à coluna "{column}".')

