class FeatureImportanceReport:
    def generate(self, plan: list) -> list:
        report = []

        for step in plan:
            column = step["column"]
            action = step["suggestion"]
            stats = step.get("parameters", {})
            description = self._describe_action(action, column)

            report.append({
                "column": column,
                "problem": "low_feature_importance",
                "action_taken": action,
                "description": description,
                "statistics": stats
            })

        return report

    def _describe_action(self, action: str, column: str) -> str:
        desc_map = {
            "REMOVER_VARIAVEL": f'Coluna "{column}" removida por baixa importância no modelo.',
            "LIMITE_TOP_N": f'Coluna "{column}" removida por estar fora do ranking de variáveis mais importantes.',
            "NENHUMA": f'Nenhuma alteração aplicada à coluna "{column}".'
        }
        return desc_map.get(action, f'Ação "{action}" aplicada à coluna "{column}".')

# Exemplo de uso:
# reporter = FeatureImportanceReport()
# feature_importance_report = reporter.generate(feature_importance_plan)
# feature_importance_report[:5]
