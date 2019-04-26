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
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import make_scorer, roc_auc_score

# For testing
rand_state = 0

# Load cleaned data and filter down to what exists in OCon
df_mid = pd.read_csv("../Data/DYDMID3.1/mid-clean.txt", sep="\t")
df_mid = df_mid.loc[df_mid["IN_ORIG"]]

# load model data for model parameters later
fcp = FullContextProcessor("../Data/OConnor2013/ocon-nicepaths-month-indexed.txt", sep="\t")

# Get train - test split 80/20 
train_idxs, test_idxs, _, _ = train_test_split(
    np.arange(df_mid.shape[0]), df_mid["HOST"].values, test_size=0.20, 
    shuffle=True, random_state=rand_state)
y = df_mid["HOST"].values

# For each model, do 5-folds cv, use best for eval and record evald ROC_AUC
model_alphas = ["1.00E-01", "1.00E-02", "1.00E-03", "1.00E-04", "1.00E-05"]
logreg = LogisticRegression(penalty="l2", solver="saga", max_iter=1000)
results = []
for alpha in model_alphas:
    # Load model and gets the embeddings to use as features
    model = SRCTSoftmaxModel(s_cnt=len(fcp.df["SOURCE_IDX"].unique()),
                             r_cnt=len(fcp.df["RECEIVER_IDX"].unique()),
                             p_cnt=len(fcp.df["PRED_IDX"].unique()),
                             T=len(fcp.df["TIME"].unique()),
                             K_s=150,
                             K_r=150,
                             K_p=300,)

    model.load_state_dict(torch.load(
        "month_softmax_K300_lr1.00E+00_lam0.00E+00_alpha{}_bs32_epochs50.pt".format(alpha),
        map_location="cpu"))
    s_embeds = model.s_embeds.weight.detach().cpu().numpy()
    r_embeds = model.r_embeds.weight.detach().cpu().numpy()

    # Create dataset in matrix representation
    X_train = np.zeros((train_idxs.shape[0], model.p_embeds.weight.shape[1]))
    y_train = y[train_idxs]
    for i, row in df_mid.iloc[train_idxs].reset_index(drop=True).iterrows():
        X_train[i, :] = np.concatenate((
            s_embeds[row["SOURCE_IDX"] + row["TIME"]*model.s_cnt],
            r_embeds[row["RECEIVER_IDX"] + row["TIME"]*model.r_cnt]))

    # Perform 5-fold cv based on roc_auc
    param_grid = {
        "C":[1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8],
    }
    gs = GridSearchCV(estimator=logreg,
                      param_grid=param_grid,
                      scoring={"roc_auc":make_scorer(roc_auc_score, needs_threshold=True)},
                      n_jobs=-1,
                      cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=rand_state),
                      refit="roc_auc",
                      verbose=30,
                      return_train_score=True,)
    gs.fit(X_train, y_train)

    # Use best lambda/C to eval on test set
    X_test = np.zeros((test_idxs.shape[0], model.p_embeds.weight.shape[1]))
    y_test = y[test_idxs]
    for i, row in df_mid.iloc[test_idxs].reset_index(drop=True).iterrows():
        X_test[i, :] = np.concatenate((
            s_embeds[row["SOURCE_IDX"] + row["TIME"]*model.s_cnt],
            r_embeds[row["RECEIVER_IDX"] + row["TIME"]*model.r_cnt]))
    test_score = gs.score(X_test, y_test) 
    print("alpha: {} | best_C: {} | test_score_roc_auc: {}".format(
        alpha, gs.best_params_["C"], test_score))

    # Log results
    for i in range(len(param_grid["C"])):
        results.append({
            "alpha":alpha,
            "lam":1.0/gs.cv_results_["params"][i]["C"],
            "mean_train_roc_auc":gs.cv_results_["mean_train_roc_auc"][i],
            "mean_eval_roc_auc":gs.cv_results_["mean_test_roc_auc"][i],
            "best_lambda":1.0/gs.best_params_["C"],
            "test_roc_auc":test_score,
        })

# Save results as a dataframe
df_results = pd.DataFrame(results)
df_results.to_csv("mids-logreg-results.txt", sep="\t", index=False)
