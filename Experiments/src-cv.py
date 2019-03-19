import sys, os
# Adding Research directory to path
sys.path.append(os.path.dirname(sys.path[0]))

# Import personal modules
from Models.SGNS import SourceReceiverConcatClassifier, SourceReceiverConcatModel
from Preprocessing.FullContextProcessor import FullContextProcessor

# Import installed modules
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import make_scorer, log_loss, f1_score, precision_score, recall_score, accuracy_score
import torch
import numpy as np
import pandas as pd

fcp = FullContextProcessor("../Data/OConnor2013/ocon-verb-noun-extracted.txt", "\t")

fcp.createTwoWayMap("SOURCE")
fcp.createTwoWayMap("RECEIVER")
fcp.createTwoWayMap("WORD")
fcp.convertColToIdx("SOURCE")
fcp.convertColToIdx("RECEIVER")
fcp.convertColToIdx("WORD")

X = fcp.df.loc[:, ["SOURCE", "RECEIVER", "WORD"]].values
y = np.ones(X.shape[0])

k = 20
neg_word_idxs = fcp.returnNumpyNegSamples(k=k,
                                          alpha=0.75,
                                          colname="WORD",
                                          seed=0)
X_neg = np.tile(X[:, :2], (k, 1))
X_neg = np.concatenate((X_neg, neg_word_idxs.reshape(-1, 1)), axis=1)
y_neg = np.zeros(X_neg.shape[0])

X = np.concatenate((X, X_neg), axis=0)
y = np.concatenate((y, y_neg), axis=0)

# #TODO GET RID OF TEST
# X = X[:100, :]
# y = y[:100]
# y[-90:] = 0.0

# Get train, val, test splits
X, X_test, y, y_test = train_test_split(X, 
                                        y, 
                                        test_size=0.15, 
                                        shuffle=True, 
                                        stratify=y)

_, _, train_idxs, test_idxs = train_test_split(X, 
                                               np.arange(y.size),
                                               test_size=0.20, 
                                               shuffle=True, 
                                               stratify=y)

# Set up cross validation, want to evaluate on BCE
src_class = SourceReceiverConcatClassifier(s_cnt=len(fcp.df["SOURCE"].unique()),
                                    r_cnt=len(fcp.df["RECEIVER"].unique()),
                                    w_cnt=len(fcp.df["WORD"].unique()),
                                    train_epocs = 1,
                                    log_fpath = "./logs/src-cv.log")

scoring = {
    "Log-Loss": make_scorer(log_loss, greater_is_better=False),
    "Accuracy": make_scorer(accuracy_score),
    "Precision": make_scorer(precision_score),
    "Recall": make_scorer(recall_score),
    "F1": make_scorer(f1_score),
}
param_grid = {
    "K":[100, 200],
    "lr":[1e0, 5e-1, 1e-1],
    "weight_decay":[0.0, 1e-3, 1e-6],
    "batch_size":[1, 16, 32],
}

gs = GridSearchCV(estimator=src_class,
                  param_grid=param_grid,
                  scoring=scoring,
                  n_jobs=1,
                  cv=[(train_idxs, test_idxs)],
                  refit="Log-Loss",
                  verbose=30,
                  return_train_score=True,)
gs.fit(X, y)

# Best model is automatically retrained, now get test performance
y_pred = gs.predict(X_test)
print("test logloss: {} | Acc: {} | Prec: {} | Rec: {} | test F1: {}".\
    format(log_loss(y_test, y_pred), 
           accuracy_score(y_test, y_pred),
           precision_score(y_test, y_pred), 
           recall_score(y_test, y_pred),
           f1_score(y_test, y_pred)),)


# Save cv results
pd.DataFrame(gs.cv_results_).to_csv(path_or_buf="src-cv-results.txt", sep="\t")

