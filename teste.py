import pandas as pd
from src.analyzers.collect.collect_group import CollectGroup
from src.analyzers.missing_values.missing_values_group import MissingGroup
from src.analyzers.distribution.distribution_group import DistributionGroup
from src.analyzers.correlation.correlation_group import CorrelationGroup
from src.analyzers.outlier.outliers_group import OutlierGroup
from src.analyzers.normalization.normalization_group import NormalizationGroup
from src.analyzers.categorical.categorical_group import CategoricalGroup
from src.analyzers.dimensionalit.dimensionalit_group import DimensionalityGroup
from src.analyzers.low_variability.low_variability_group import LowVariabilityGroup




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
    
    outlier_group = OutlierGroup()
    outlier_result = outlier_group.run(correlation_result["data"])
    
    normalization_group = NormalizationGroup()  
    normalization_result = normalization_group.run(outlier_result["data"])

    categorical_group = CategoricalGroup()
    categorical_result = categorical_group.run(normalization_result["data"])
    
    dimensionality_group = DimensionalityGroup()
    dimensionality_result = dimensionality_group.run(categorical_result["data"])    
    
    low_variability_group = LowVariabilityGroup()
    low_variability_result = low_variability_group.run(dimensionality_result["data"])
    print(low_variability_result)


if __name__ == "__main__":
    main()