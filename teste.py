import pandas as pd
from src.analyzers.collect.collect_group import CollectGroup
from src.analyzers.missing_values.missing_values_group import MissingGroup



def main():
    # Carregando o dataset
    data = pd.read_csv("dados/train.csv")

    collect_group = CollectGroup()
    result = collect_group.run(data)
    
    missing_group = MissingGroup()
    result = missing_group.run(result["data"])
    print(result)
    


if __name__ == "__main__":
    main()