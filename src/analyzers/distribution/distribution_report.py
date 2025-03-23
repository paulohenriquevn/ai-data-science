from src.analyzers.base.analysis_base import ReportStep

class DistributionReport(ReportStep):
    def generate(self, plan: list) -> list:
        report = []

        for step in plan:
            column = step['column']
            action = step['suggestion']
            description = self._describe_action(action, column)
            stats = step.get('parameters', {})

            report.append({
                'column': column,
                'problem': 'distribution_pattern',
                'action_taken': action,
                'description': description,
                'statistics': stats
            })

        return report

    def _describe_action(self, action: str, column: str) -> str:
        desc_map = {
            'TRANSFORMACAO_LOG': f'Transformação logarítmica aplicada à coluna "{column}".',
            'TRANSFORMACAO_RAIZ_QUADRADA': f'Transformação de raiz quadrada aplicada à coluna "{column}".',
            'TRANSFORMACAO_BOX_COX': f'Transformação Box-Cox aplicada à coluna "{column}".',
            'TRANSFORMACAO_REFLECAO_LOG': f'Transformação logarítmica com reflexão aplicada à coluna "{column}".',
            'TRANSFORMACAO_QUADRATICA': f'Transformação quadrática aplicada à coluna "{column}".',
            'TRANSFORMACAO_INVERSA': f'Transformação inversa aplicada à coluna "{column}".',
            'CENTRALIZACAO_PADRONIZACAO': f'Centralização e padronização aplicadas à coluna "{column}".',
            'WINSORIZATION': f'Winsorization aplicada à coluna "{column}".',
            'CLIPPING': f'Clipping aplicado à coluna "{column}".',
            'ALGORITMO_ROBUSTO': f'Coluna "{column}" marcada para tratamento com algoritmo robusto.',
            'NENHUMA': f'Nenhuma transformação aplicada à coluna "{column}".'
        }
        return desc_map.get(action, f'Ação "{action}" aplicada à coluna "{column}".')


