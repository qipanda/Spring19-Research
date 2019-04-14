# Adding Research directory to path and parse script arguments
import sys, os, json
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))

# Import custom modules
from Preprocessing.FullContextProcessor import FullContextProcessor

# Import installed modules
import pandas as pd

with open("../../Data/Westveld-Hoff2011/country-dict.json", "r") as f:
    country_dict = json.load(f)

def strReplaceDict(string):
    for old, new in country_dict.items():
        string = string.replace(old, new)

    return string

# Clean the trade data
df = pd.read_csv("../../Data/Westveld-Hoff2011/Trade.csv")
df = df.iloc[:, 1]
df = df.apply(strReplaceDict)
df = df.str.split(" ", expand=True)
df = df.iloc[:, 3:7]
df.columns = ["YEAR", "SOURCE", "RECEIVER", "LN_TRADE"]
df.loc[:, "YEAR"] = df["YEAR"].apply(lambda x: int(x)+1980)
df.loc[:, "LN_TRADE"] = df["LN_TRADE"].astype(float)

# Save for later use
df.to_csv("../../Data/Westveld-Hoff2011/sr-trade.txt", sep="\t", index=False)
