class RareCategoryReport:
    def generate(self, plan: list) -> list:
        report = []

        for step in plan:
            column = step["column"]
            action = step["suggestion"]
            stats = step.get("parameters", {})
            description = self._describe_action(action, column)

            report.append({
                'column': column,
                'problem': 'rare_category_detection',
                'action_taken': action,
                'description': description,
                'statistics': stats
            })

        return report

    def _describe_action(self, action: str, column: str) -> str:
        desc_map = {
            "AGRUPAR_RARAS": f'Categorias raras da coluna "{column}" foram agrupadas em "OUTROS".',
            "REDUZIR_CARDINALIDADE": f'Coluna "{column}" teve sua cardinalidade reduzida: mantidas apenas as categorias mais frequentes.',
            "NENHUMA": f'Nenhuma modificação realizada na coluna "{column}".'
        }
        return desc_map.get(action, f'Ação "{action}" aplicada à coluna "{column}".')
