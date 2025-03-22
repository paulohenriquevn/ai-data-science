from enum import Enum
from typing import Dict, Any
import pandas as pd
from src.analyzers.base.analysis_base import AnalysisStep


class BalanceProblem(Enum):
    BALANCEADO = "Balanceamento adequado"
    DESEQUILIBRIO_LEVE = "Desequilíbrio leve"
    DESEQUILIBRIO_MODERADO = "Desequilíbrio moderado"
    DESEQUILIBRIO_FORTE = "Desequilíbrio forte"


class BalanceSolution(Enum):
    NENHUMA = "Nenhuma ação necessária"
    UNDERSAMPLING = "Aplicar undersampling na classe majoritária"
    OVERSAMPLING = "Aplicar oversampling na classe minoritária"
    SMOTE = "Aplicar técnica SMOTE para balanceamento"
    AUMENTAR_DADOS = "Coletar ou gerar mais dados da classe minoritária"


class BalanceAnalyzer(AnalysisStep):
    def __init__(self, target_column: str = "y"):
        self.target_column = target_column

    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        if self.target_column not in data.columns:
            raise ValueError(f"Coluna alvo '{self.target_column}' não encontrada no DataFrame.")

        class_counts = data[self.target_column].value_counts().to_dict()
        total = sum(class_counts.values())
        majority = max(class_counts.values())
        minority = min(class_counts.values())
        imbalance_ratio = minority / majority if majority != 0 else 0

        if imbalance_ratio > 0.8:
            problem = BalanceProblem.BALANCEADO
            solutions = [BalanceSolution.NENHUMA]
        elif imbalance_ratio > 0.6:
            problem = BalanceProblem.DESEQUILIBRIO_LEVE
            solutions = [BalanceSolution.SMOTE, BalanceSolution.OVERSAMPLING]
        elif imbalance_ratio > 0.4:
            problem = BalanceProblem.DESEQUILIBRIO_MODERADO
            solutions = [BalanceSolution.SMOTE, BalanceSolution.OVERSAMPLING, BalanceSolution.UNDERSAMPLING]
        else:
            problem = BalanceProblem.DESEQUILIBRIO_FORTE
            solutions = [BalanceSolution.SMOTE, BalanceSolution.OVERSAMPLING, BalanceSolution.UNDERSAMPLING, BalanceSolution.AUMENTAR_DADOS]

        return {
            "target_column": self.target_column,
            "class_distribution": class_counts,
            "imbalance_ratio": round(imbalance_ratio, 3),
            "problem": problem.name,
            "problem_description": problem.value,
            "solution": solutions[0].name,
            "actions": [s.name for s in solutions]
        }


