class ResidualCollinearityReport:
    def generate(self, plan: list) -> list:
        report = []

        for step in plan:
            column = step["column"]
            action = step["suggestion"]
            stats = step.get("parameters", {})
            description = self._describe_action(action, column)

            report.append({
                "column": column,
                "problem": "residual_collinearity",
                "action_taken": action,
                "description": description,
                "statistics": stats
            })

        return report

    def _describe_action(self, action: str, column: str) -> str:
        desc_map = {
            "REMOVER_VARIAVEL": f'Coluna "{column}" removida devido a alto VIF.',
            "REMOVER_REDUNDANTE": f'Coluna "{column}" removida por correlação alta com outra variável.',
            "EXCLUIR_POR_RELEVANCIA_BAIXA": f'Coluna "{column}" removida por ser colinear e ter baixa correlação com o alvo.',
            "NENHUMA": f'Nenhuma modificação aplicada à coluna "{column}".'
        }
        return desc_map.get(action, f'Ação "{action}" aplicada à coluna "{column}".')

# Exemplo de uso:
# reporter = ResidualCollinearityReport()
# collinearity_report = reporter.generate(collinearity_plan)
# collinearity_report[:5]
