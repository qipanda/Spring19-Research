# Adding Research directory to path and parse script arguments
import sys, os, json
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))

# Import custom modules
from Preprocessing.FullContextProcessor import FullContextProcessor

# Import installed modules
import pandas as pd

def cartesian_product(left, right):
    return (
       left.assign(key=1).merge(right.assign(key=1), on='key').drop('key', 1))

# Import COW country code conversion to ocon country codes
df_cow = pd.read_csv("../../Data/DYDMID3.1/COW_country_codes.csv")
dict_cow = df_cow.loc[~df_cow["OConAbb"].isna()]\
    .loc[:, ["StateAbb", "OConAbb"]]\
    .set_index("StateAbb")\
    .to_dict()["OConAbb"]

# Clean the trade data to match OCon data
hostile_phrases = {"Use of Force", "Interstate War"}
df = pd.io.stata.read_stata("../../Data/DYDMID3.1/dyadic_mid_3.1.dta")
df = df.loc[df["hihost"].apply(lambda x: x in hostile_phrases),
    ["namea", "nameb", "strtmnth", "strtyr", "endmnth", "endyear"]]
df.loc[:, "namea"] = df["namea"].apply(lambda x: dict_cow[x] if x in dict_cow else x)
df.loc[:, "nameb"] = df["nameb"].apply(lambda x: dict_cow[x] if x in dict_cow else x)

# Explode date ranges into all months
df_exp = pd.DataFrame(columns=["SOURCE", "RECEIVER", "YEAR", "MONTH"])
for i, row in df.iterrows():
    endyear, endmonth = row["endyear"], row["endmnth"]
    startyear, startmonth = row["strtyr"], row["strtmnth"]

    if endyear > 2008:
        endyear=2008
        endmonth=12
    if startyear < 1987:
        startyear = 1987
        startmonth = 1

    while startyear*100 + startmonth <= endyear*100 + endmonth:
        df_exp = df_exp.append({
            "SOURCE":row["namea"],
            "RECEIVER":row["nameb"],
            "YEAR":startyear,
            "MONTH":startmonth,
        }, ignore_index=True)

        if startmonth == 12:
            startyear += 1
            startmonth = 1
        else:
            startmonth += 1
df_exp = df_exp.drop_duplicates()
df_exp["HOST"] = True

# Load Ocon data
fcp = FullContextProcessor("../../Data/OConnor2013/ocon-nicepaths-month-indexed.txt", "\t")

# Find common sources and receivers so can see what to train/predict on
s_common = set(df_exp["SOURCE"].unique()).intersection(set(fcp.df["SOURCE"].unique()))
r_common = set(df_exp["RECEIVER"].unique()).intersection(set(fcp.df["RECEIVER"].unique()))

s_common = fcp.df.loc[fcp.df["SOURCE"].isin(s_common), ["SOURCE", "SOURCE_IDX"]]\
           .drop_duplicates()\
           .reset_index(drop=True)
r_common = fcp.df.loc[fcp.df["RECEIVER"].isin(r_common), ["RECEIVER", "RECEIVER_IDX"]]\
           .drop_duplicates()\
           .reset_index(drop=True)
year_months = fcp.df.loc[:, ["YEAR", "MONTH", "TIME"]]\
        .drop_duplicates()\
        .reset_index(drop=True)

# Get all combinations of s_common, r_common, and years (exist in Hoff data + our model params)
df_cart = cartesian_product(cartesian_product(s_common, r_common), year_months)
df_cart = df_cart.merge(df_exp, on=["SOURCE", "RECEIVER", "YEAR", "MONTH"], how="left")
df_cart.loc[df_cart["HOST"].isna(), "HOST"] = False

# Remove any (s,r) rows where the whole (s,r) has not a single True "HOST"
df_cart = df_cart.groupby(["SOURCE", "RECEIVER"]).filter(lambda g: g["HOST"].sum() > 0)

# Categorize rows as either existing in corpus data
df_corpus = fcp.df.loc[:, ["SOURCE", "RECEIVER", "TIME"]].drop_duplicates()
df_corpus["IN_ORIG"] = True
df_cart = df_cart.merge(df_corpus, on=["SOURCE", "RECEIVER", "TIME"], how="left")
df_cart.loc[df_cart["IN_ORIG"].isna(), "IN_ORIG"] = False

# Save for later use
df_cart.to_csv("../../Data/DYDMID3.1/mid-clean.txt", sep="\t", index=False)
