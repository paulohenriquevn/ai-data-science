class NormalizationReport:
    def generate(self, plan: list) -> list:
        report = []

        for step in plan:
            column = step['column']
            action = step['suggestion']
            stats = step.get('parameters', {})
            description = self._describe_action(action, column)

            report.append({
                'column': column,
                'problem': 'scaling_needed',
                'action_taken': action,
                'description': description,
                'statistics': stats
            })

        return report

    def _describe_action(self, action: str, column: str) -> str:
        desc_map = {
            'STANDARD_SCALER': f'Aplicado StandardScaler na coluna "{column}".',
            'ROBUST_SCALER': f'Aplicado RobustScaler na coluna "{column}".',
            'NENHUMA': f'Nenhuma transformação aplicada à coluna "{column}".'
        }
        return desc_map.get(action, f'Ação "{action}" aplicada à coluna "{column}".')
