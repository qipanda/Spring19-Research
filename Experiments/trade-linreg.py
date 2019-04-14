# Adding Research directory to path and parse script arguments
import sys, os, argparse, json
sys.path.append(os.path.dirname(sys.path[0]))

# Import custom modules
from Preprocessing.FullContextProcessor import FullContextProcessor
from Models.models import SRCTModel

# Import installed modules
import pandas as pd
import numpy as np
import torch
import plotly
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def cartesian_product(left, right):
    return (
       left.assign(key=1).merge(right.assign(key=1), on='key').drop('key', 1))

df_trade = pd.read_csv("../Data/Westveld-Hoff2011/sr-trade.txt", sep="\t")

# load sgns data
fcp = FullContextProcessor("../Data/OConnor2013/ocon-nicepaths-year-indexed.txt", sep="\t")

# Find common sources and receivers so can see what to train/predict on
s_common = set(df_trade["SOURCE"].unique()).intersection(set(fcp.df["SOURCE"].unique()))
r_common = set(df_trade["RECEIVER"].unique()).intersection(set(fcp.df["RECEIVER"].unique()))

s_common = fcp.df.loc[fcp.df["SOURCE"].isin(s_common), ["SOURCE", "SOURCE_IDX"]]\
           .drop_duplicates()\
           .reset_index(drop=True)
r_common = fcp.df.loc[fcp.df["RECEIVER"].isin(r_common), ["RECEIVER", "RECEIVER_IDX"]]\
           .drop_duplicates()\
           .reset_index(drop=True)
years = fcp.df.loc[(fcp.df["YEAR"] <= 2000) & (fcp.df["YEAR"] >= 1981), ["YEAR", "TIME"]]\
        .drop_duplicates()\
        .reset_index(drop=True)

# Get all combinations of s_common, r_common, and years (exist in Hoff data + our model params)
df_cart = cartesian_product(cartesian_product(s_common, r_common), years)
df_cart = df_cart.merge(df_trade, on=["SOURCE", "RECEIVER", "YEAR"])

# Categorize rows as either existing in corpus data
df_corpus = fcp.df.loc[:, ["SOURCE", "RECEIVER", "YEAR"]].drop_duplicates()
df_corpus["IN_ORIG"] = True
df_cart = df_cart.merge(df_corpus, on=["SOURCE", "RECEIVER", "YEAR"], how="left")
df_cart.loc[df_cart["IN_ORIG"].isna(), "IN_ORIG"] = False

# Load model to use as features
model = SRCTModel(s_cnt=len(fcp.df["SOURCE_IDX"].unique()),
                  r_cnt=len(fcp.df["RECEIVER_IDX"].unique()),
                  p_cnt=len(fcp.df["PRED_IDX"].unique()),
                  T=len(fcp.df["TIME"].unique()),
                  K_s=100,
                  K_r=100,
                  K_p=200,)

model.load_state_dict(torch.load(
    "K200_lr1.00E+00_lam0.00E+00_alpha1.00E-04_bs32_epochs10.pt",
    map_location="cpu"))

# Create the dataset of SR embeddings
X = np.empty((df_cart.shape[0], model.p_embeds.weight.shape[1]))
y = np.empty(df_cart.shape[0])
s_embeds = model.s_embeds.weight.detach().numpy()
r_embeds = model.r_embeds.weight.detach().numpy()
for i, row in df_cart.iterrows():
    X[i, :] = np.concatenate((
        s_embeds[row["SOURCE_IDX"] + row["TIME"]*model.s_cnt],
        r_embeds[row["RECEIVER_IDX"] + row["TIME"]*model.r_cnt]))
    y[i] = row["LN_TRADE"]

# Randomly select 75% for training a linear regression model, predict on 25% and get MSE
# Repeat many times with different splits to account for variance in sample size
trials = int(1)
mses = np.zeros(trials)
reg = LinearRegression()
for i in range(trials):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True)
    reg.fit(X_train, y_train)
    y_preds = reg.predict(X_test)
    mses[i] = mean_squared_error(y_true=y_test, y_pred=y_preds)

import ipdb; ipdb.set_trace()

# TODO compare with Westveld-Hoff model on same train-test splitting
print("MSE: {}".format(np.mean(mses)))
# Plot histogram of mse's
data = [go.Histogram(x=mses)]
plotly.offline.plot(data, filename="mse_hist_lin_reg.html")

# Plot projections onto the line as x and the ln(trade) as y seperately for train and test
W = reg.coef_
X_train_plot = np.linalg.norm(X_train*W, ord=2.0, axis=1)
y_train_plot = y_train - reg.predict(X_train)
train_trace = go.Scatter(x=X_train_plot, y=y_train_plot, mode="markers", name="train")

X_test_plot = np.linalg.norm(X_test*W, ord=2.0, axis=1)
y_test_plot = y_test - y_preds
test_trace = go.Scatter(x=X_test_plot, y=y_test_plot, mode="markers", name="test")

data = [train_trace, test_trace]
plotly.offline.plot(data, filename="projected_lin_reg.html")

