import pandas as pd
from src.analyzers.collect.collect_group import CollectGroup

def main():
    # Carregando o dataset
    data = pd.read_csv("dados/train.csv")

    collect_group = CollectGroup()
    result = collect_group.run(data)

    print(result)


if __name__ == "__main__":
    main()