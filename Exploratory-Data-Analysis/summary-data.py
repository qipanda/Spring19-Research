import pandas as pd

data = pd.read_csv("../Data/abcnews-clean.txt", sep="\t")
print(data.groupby(["c1", "c2"])["date"].count().sort_values())
