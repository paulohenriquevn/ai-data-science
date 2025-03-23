class FeatureInteractionReport:
    def generate(self, plan: list) -> list:
        report = []

        for step in plan:
            col1, col2 = step['column']
            action = step['suggestion']
            stats = step.get('parameters', {})
            description = self._describe_action(action, col1, col2)

            report.append({
                'column': [col1, col2],
                'problem': 'potential_interaction',
                'action_taken': action,
                'description': description,
                'statistics': stats
            })

        return report

    def _describe_action(self, action: str, col1: str, col2: str) -> str:
        desc_map = {
            "CRIAR_INTERACAO_PRODUTO": f'Criada nova feature como produto entre "{col1}" e "{col2}".',
            "CRIAR_RAZAO": f'Criada nova feature como razão entre "{col1}" e "{col2}".',
            "CRIAR_DIFERENCA": f'Criada nova feature como diferença entre "{col1}" e "{col2}".',
            "NENHUMA": f'Nenhuma interação criada entre "{col1}" e "{col2}".'
        }
        return desc_map.get(action, f'Interação entre "{col1}" e "{col2}" com ação "{action}".')

