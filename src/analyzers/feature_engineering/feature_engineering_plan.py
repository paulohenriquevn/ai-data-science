from src.analyzers.base.analysis_base import PlanStep

class FeatureEngineeringPlan(PlanStep):
    def __init__(self, distribution_report=None, outlier_report=None, significance_report=None, correlation_report=None):
        self.distribution_report = distribution_report or []
        self.outlier_report = outlier_report or []
        self.significance_report = significance_report or []
        self.correlation_report = correlation_report or []

    def generate(self):
        transformations = {}
        flags = []
        interactions = []
        exclude_from_scaling = ["id", "y"]

        # Baseado na distribuição
        for item in self.distribution_report:
            col = item.get("column")
            problem = item.get("problem", "")
            stats = item.get("statistics", {})
            skew = stats.get("skewness", 0)

            if problem == "ASSIMETRICA_POSITIVA" or skew > 1:
                transformations[col] = "sqrt"
            elif problem == "ASSIMETRICA_NEGATIVA" or skew < -1:
                transformations[col] = "square"

        # Baseado em valores ausentes/extremos (outlier report)
        for item in self.outlier_report:
            col = item.get("column")
            stats = item.get("statistics", {})
            outlier_ratio = stats.get("outlier_ratio", 0)
            if outlier_ratio > 0.4:
                flags.append(col)

        # Baseado em significância estatística + correlação
        variaveis_significativas = set()
        variaveis_correlacionadas = set()

        for item in self.significance_report:
            col = item.get("column")
            p = item.get("statistics", {}).get("p_value")
            if p is not None and p < 0.05:
                variaveis_significativas.add(col)

        for item in self.correlation_report:
            col = item.get("column")
            strength = item.get("statistics", {}).get("correlation_strength")
            if strength in ["MODERADA", "FORTE"]:
                variaveis_correlacionadas.add(col)

        # Criar interações entre variáveis que são relevantes e correlacionadas
        intersecao = list(variaveis_significativas & variaveis_correlacionadas)
        for i in range(len(intersecao)):
            for j in range(i + 1, len(intersecao)):
                interactions.append((intersecao[i], intersecao[j]))

        return {
            "transformations": transformations,
            "flags": flags,
            "interactions": interactions,
            "exclude_from_scaling": exclude_from_scaling
        }
