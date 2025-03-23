import pandas as pd
from src.analyzers.collect.collect_group import CollectGroup
from src.analyzers.missing_values.missing_values_group import MissingGroup
from src.analyzers.distribution.distribution_group import DistributionGroup
from src.analyzers.correlation.correlation_group import CorrelationGroup
from src.analyzers.outlier.outliers_analyzer import OutlierAnalyzer
from src.analyzers.outlier.outliers_plan import OutlierPlan
from src.analyzers.outlier.outliers_executor import OutlierExecutor
from src.analyzers.outlier.outliers_report import OutlierReport
from src.analyzers.outlier.outliers_group import OutlierGroup





def main():
    # Carregando o dataset
    data = pd.read_csv("dados/train.csv")

    collect_group = CollectGroup()
    result = collect_group.run(data)
    
    missing_group = MissingGroup()
    result = missing_group.run(result["data"])
    
    distribution_group = DistributionGroup()
    distribution_result = distribution_group.run(result["data"])
    
    correlation_group = CorrelationGroup(target_column='y')
    correlation_result = correlation_group.run(distribution_result["data"])
    
    # Rodar análise de outliers no DataFrame
    outlier_analyzer = OutlierAnalyzer()
    outlier_analysis_result = outlier_analyzer.analyze(correlation_result["data"])

    # Gerar plano a partir do diagnóstico de outliers
    outlier_planner = OutlierPlan()
    outlier_plan = outlier_planner.generate(outlier_analysis_result)
    
    # Aplicar executor no DataFrame original
    outlier_executor = OutlierExecutor()
    df_outlier_cleaned = outlier_executor.execute(correlation_result["data"], outlier_plan)

    # Gerar o relatório de outliers
    outlier_reporter = OutlierReport()
    outlier_report = outlier_reporter.generate(outlier_plan)

    # Executar o grupo de outlier
    outlier_group = OutlierGroup()
    outlier_result = outlier_group.run(correlation_result["data"])

    print(outlier_result)


if __name__ == "__main__":
    main()