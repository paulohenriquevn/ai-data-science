import pandas as pd
from src.analyzers.collect.collect_group import CollectGroup
from src.analyzers.missing_values.missing_values_group import MissingGroup
from src.analyzers.distribution.distribution_analyzer import DistributionAnalyzer
from src.analyzers.distribution.distribution_plan import DistributionPlan
from src.analyzers.distribution.distribution_executor import DistributionExecutor
from src.analyzers.distribution.distribution_report import DistributionReport
from src.analyzers.distribution.distribution_group import DistributionGroup


def main():
    # Carregando o dataset
    data = pd.read_csv("dados/train.csv")

    collect_group = CollectGroup()
    result = collect_group.run(data)
    
    missing_group = MissingGroup()
    result = missing_group.run(result["data"])
    print(result)
    
    distribution_group = DistributionGroup()
    distribution_result = distribution_group.run(result["data"])
    print(distribution_result)
    


if __name__ == "__main__":
    main()