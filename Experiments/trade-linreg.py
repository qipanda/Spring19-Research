# Adding Research directory to path and parse script arguments
import sys, os, argparse, json
sys.path.append(os.path.dirname(sys.path[0]))

# Import custom modules
from Preprocessing.FullContextProcessor import FullContextProcessor
from Models.models import SRCTModel, SRCTSoftmaxModel

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

# Load various softmax models to use as features
model_alphas = ["1.00E-01", "1.00E-02", "1.00E-03", "1.00E-04", "1.00E-05"]
embeds = {}
for alpha in model_alphas:
    model = SRCTSoftmaxModel(s_cnt=len(fcp.df["SOURCE_IDX"].unique()),
                             r_cnt=len(fcp.df["RECEIVER_IDX"].unique()),
                             p_cnt=len(fcp.df["PRED_IDX"].unique()),
                             T=len(fcp.df["TIME"].unique()),
                             K_s=150,
                             K_r=150,
                             K_p=300,)

    model.load_state_dict(torch.load(
        "year_softmax_K300_lr1.00E+00_lam0.00E+00_alpha{}_bs32_epochs50.pt".format(alpha),
        map_location="cpu"))

    embeds[alpha] = {
        "s_embeds":model.s_embeds.weight.detach().numpy(),
        "r_embeds":model.r_embeds.weight.detach().numpy(),
        "X":np.zeros((df_cart.shape[0], model.p_embeds.weight.shape[1]))
    }

# Create features for Westveld-Hoff 2011 model and srct-model
X_westhoff = df_cart.loc[:, ["S_LN_GDP", "R_LN_GDP", "LN_DIST", "S_POL", "R_POL", "CC"]]
X_westhoff["S_POL X R_POL"] = X_westhoff["S_POL"] * X_westhoff["R_POL"]
X_westhoff = X_westhoff.values
y = np.empty(df_cart.shape[0])

for i, row in df_cart.iterrows():
    X_westhoff[i, :] = np.array([
        row["S_LN_GDP"],
        row["R_LN_GDP"],
        row["LN_DIST"],
        row["S_POL"],
        row["R_POL"],
        row["CC"],
        row["S_POL"]*row["R_POL"]])
    for alpha in embeds.keys():
        embeds[alpha]["X"][i, :] = np.concatenate((
            embeds[alpha]["s_embeds"][row["SOURCE_IDX"] + row["TIME"]*model.s_cnt],
            embeds[alpha]["r_embeds"][row["RECEIVER_IDX"] + row["TIME"]*model.r_cnt]))

    y[i] = row["LN_TRADE"]

# Randomly select 75% for training a linear regression model, predict on 25% and get MSE
# Repeat many times with different splits to account for variance in sample size
trials = int(1000)
mses = {}
for alpha in model_alphas:
    mses[alpha] = np.zeros(trials)
mses["westhoff"] = np.zeros(trials)
mses["baseline_mean"] = np.zeros(trials)

reg = LinearRegression(normalize=True)
for i in range(trials):
    train_idxs, test_idxs, _, _ = train_test_split(
        np.arange(df_cart.shape[0]), y, test_size=0.25, shuffle=True)

    for alpha in model_alphas:
        reg.fit(embeds[alpha]["X"][train_idxs], y[train_idxs])
        y_preds = reg.predict(embeds[alpha]["X"][test_idxs])
        mses[alpha][i] = mean_squared_error(y_true=y[test_idxs], y_pred=y_preds)

    reg.fit(X_westhoff[train_idxs], y[train_idxs])
    y_preds = reg.predict(X_westhoff[test_idxs])
    mses["westhoff"][i] = mean_squared_error(y_true=y[test_idxs], y_pred=y_preds)

    y_preds = np.ones(test_idxs.shape[0])*np.mean(y[train_idxs])
    mses["baseline_mean"][i] = mean_squared_error(y_true=y[test_idxs], y_pred=y_preds)

# compare with Westveld-Hoff model on same train-test splitting
data = []
for key, mse in mses.items():
    print("{} mean MSE: {}".format(key, np.mean(mse)))
    data.append(go.Histogram(x=mse, name=key))

# Plot histogram of mse's
plotly.offline.plot(data, filename="mse_hist_lin_reg.html")

# # Plot projections onto the line as x and the ln(trade) as y seperately for train and test
# W = reg.coef_
# X_train_plot = np.linalg.norm(X_train*W, ord=2.0, axis=1)
# y_train_plot = reg.predict(X_train) - y_train
# train_trace = go.Scatter(x=X_train_plot, y=y_train_plot, mode="markers", name="train")

# X_test_plot = np.linalg.norm(X_test*W, ord=2.0, axis=1)
# y_test_plot = y_preds - y_test
# test_trace = go.Scatter(x=X_test_plot, y=y_test_plot, mode="markers", name="test")

# data = [train_trace, test_trace]
# plotly.offline.plot(data, filename="projected_lin_reg.html")

