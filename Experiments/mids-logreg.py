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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, roc_auc_score


# Load cleaned data
df_mid = pd.read_csv("../Data/DYDMID3.1/mid-clean.txt", sep="\t")

# load sgns data
fcp = FullContextProcessor("../Data/OConnor2013/ocon-nicepaths-month-indexed.txt", sep="\t")

# # Load model to use as features
# model = SRCTModel(s_cnt=len(fcp.df["SOURCE_IDX"].unique()),
#                   r_cnt=len(fcp.df["RECEIVER_IDX"].unique()),
#                   p_cnt=len(fcp.df["PRED_IDX"].unique()),
#                   T=len(fcp.df["TIME"].unique()),
#                   K_s=100,
#                   K_r=100,
#                   K_p=200,)

# model.load_state_dict(torch.load(
#     "K200_lr1.00E+00_lam0.00E+00_alpha1.00E-04_bs32_epochs10.pt",
#     map_location="cpu"))

# Load softmax model to use as features
model = SRCTSoftmaxModel(s_cnt=len(fcp.df["SOURCE_IDX"].unique()), r_cnt=len(fcp.df["RECEIVER_IDX"].unique()),
                         p_cnt=len(fcp.df["PRED_IDX"].unique()),
                         T=len(fcp.df["TIME"].unique()),
                         K_s=150,
                         K_r=150,
                         K_p=300,)

model.load_state_dict(torch.load(
    "month_softmax_K300_lr1.00E+00_lam0.00E+00_alpha5.00E-02_bs32_epochs50.pt",
    map_location="cpu"))

# Create dataset in matrix representation
X = np.zeros((df_mid.shape[0], model.p_embeds.weight.shape[1]))
y = np.empty(df_mid.shape[0])
s_embeds = model.s_embeds.weight.detach().numpy()
r_embeds = model.r_embeds.weight.detach().numpy()

for i, row in enumerate(df_mid.loc[:, ["SOURCE_IDX", "RECEIVER_IDX", "TIME", "HOST"]]\
    .itertuples(index=False)):
    X[i, :] = np.concatenate((
        s_embeds[row[0] + row[2]*model.s_cnt],
        r_embeds[row[1] + row[2]*model.r_cnt]))
    y[i] = row[3]

# Do CV then train and get test ROC
logreg = LogisticRegression()

param_grid = {
    "C":[1.0/1e4, 1.0/1e3, 1.0/1e2, 1.0/1e1, 1.0, 1.0/1e-1, 1.0/1e-2],
    "penalty":["l1"],
    "solver":["saga"],
}
scoring = {
    "ROC_AUC": make_scorer(roc_auc_score)
}
gs = GridSearchCV(estimator=logreg,
                  param_grid=param_grid,
                  scoring=scoring,
                  n_jobs=1,
                  cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=0),
                  refit=False,
                  verbose=30,
                  return_train_score=True,)
gs.fit(X, y)

# # Best model is automatically retrained, now get test performance
# y_pred = gs.predict(X_test)
# print("test logloss: {} | Acc: {} | Prec: {} | Rec: {} | test F1: {}".\
#     format(log_loss(y_test, y_pred), 
#            accuracy_score(y_test, y_pred),
#            precision_score(y_test, y_pred), 
#            recall_score(y_test, y_pred),
#            f1_score(y_test, y_pred)),)

# for i in range(trials):
#     train_idxs, test_idxs, _, _ = train_test_split(np.arange(df_cart.shape[0]), y, test_size=0.25, shuffle=True)
#     reg_srct.fit(X_srct[train_idxs], y[train_idxs])
#     reg_westhoff.fit(X_westhoff[train_idxs], y[train_idxs])
#     y_preds_srct = reg_srct.predict(X_srct[test_idxs])
#     y_preds_westhoff = reg_westhoff.predict(X_westhoff[test_idxs])
#     y_preds_mean = np.ones(test_idxs.shape[0])*np.mean(y[train_idxs])
#     mses_srct[i] = mean_squared_error(y_true=y[test_idxs], y_pred=y_preds_srct)
#     mses_westhoff[i] = mean_squared_error(y_true=y[test_idxs], y_pred=y_preds_westhoff)
#     mses_mean[i] = mean_squared_error(y_true=y[test_idxs], y_pred=y_preds_mean)

# # compare with Westveld-Hoff model on same train-test splitting
# print("mean MSE srct: {} | mean MSE westhoff: {} | mean MSE baseline: {}"\
#     .format(np.mean(mses_srct), np.mean(mses_westhoff), np.mean(mses_mean)))

# # Plot histogram of mse's
# data = [go.Histogram(x=mses_srct), go.Histogram(x=mses_westhoff), go.Histogram(x=mses_mean)]
# plotly.offline.plot(data, filename="mse_hist_lin_reg.html")
