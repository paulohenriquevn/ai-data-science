from src.analyzers.base.analysis_base import ReportStep


class MissingValuesReport(ReportStep):
    def generate(self, plan: list) -> list:
        report = []
        for step in plan:
            column = step.get('column')
            problem = step.get('problem', 'missing_values')
            action = step.get('suggestion')
            stats = step.get('parameters', {})

            description = self._describe_action(action, column)

            report.append({
                'column': column,
                'problem': problem,
                'action_taken': action,
                'description': description,
                'statistics': stats
            })

        return report

    def _describe_action(self, action: str, column: str) -> str:
        desc_map = {
            'IMPUTE_MEDIA': f'Preenchido com a média da coluna "{column}".',
            'IMPUTE_MEDIANA': f'Preenchido com a mediana da coluna "{column}".',
            'CATEGORIA_DESCONHECIDA': f'Preenchido com a categoria "Desconhecido" na coluna "{column}".',
            'ADICIONAR_FLAG_E_IMPUTAR': f'Criada flag de ausência e imputado valor padrão na coluna "{column}".',
            'EXCLUIR_LINHAS': f'Linhas com valores ausentes removidas da coluna "{column}".',
            'EXCLUIR_COLUNA': f'Coluna "{column}" foi removida do conjunto de dados.',
            'KNN_IMPUTER': f'Imputação via KNN aplicada à coluna "{column}".',
            'MICE_IMPUTER': f'Imputação multivariada (MICE) aplicada à coluna "{column}".',
            'NENHUMA_ACAO_NECESSARIA': f'Nenhuma ação necessária para a coluna "{column}".'
        }

        return desc_map.get(action, f'Ação "{action}" aplicada à coluna "{column}".')

