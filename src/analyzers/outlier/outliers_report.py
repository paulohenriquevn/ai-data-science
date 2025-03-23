class OutlierReport:
    def generate(self, plan: list) -> list:
        report = []

        for step in plan:
            column = step['column']
            action = step['suggestion']
            stats = step.get('parameters', {})
            description = self._describe_action(action, column)

            report.append({
                'column': column,
                'problem': 'outlier_detection',
                'action_taken': action,
                'description': description,
                'statistics': stats
            })

        return report

    def _describe_action(self, action: str, column: str) -> str:
        desc_map = {
            'WINSORIZATION': f'Aplicada winsorização na coluna "{column}".',
            'CLIPPING': f'Aplicado clipping (1º e 99º percentis) na coluna "{column}".',
            'TRANSFORMACAO_LOG': f'Transformação logarítmica aplicada à coluna "{column}".',
            'TRANSFORMACAO_REFLEXAO_LOG': f'Transformação logarítmica com reflexão aplicada à coluna "{column}".',
            'REMOCAO_OUTLIERS': f'Removidas linhas com outliers da coluna "{column}".',
            'NENHUMA': f'Nenhuma ação aplicada à coluna "{column}".'
        }
        return desc_map.get(action, f'Ação "{action}" aplicada à coluna "{column}".')

