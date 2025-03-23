import pandas as pd
from src.analyzers.collect.collect_group import CollectGroup
from src.analyzers.missing_values.missing_values_group import MissingGroup
from src.analyzers.distribution.distribution_group import DistributionGroup
from src.analyzers.correlation.correlation_group import CorrelationGroup




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
    
    print(correlation_result)
    

if __name__ == "__main__":
    main()