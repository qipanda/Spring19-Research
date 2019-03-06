import pickle
import sys, os
# Adding Research directory to path
sys.path.append(os.path.dirname(sys.path[0]))

# Import personal modules
from Models.SGNS import SGNSClassifier, SGNSModel
from Preprocessing.FullContextProcessor import FullContextProcessor

# Import installed modules
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import make_scorer, log_loss, f1_score
import numpy as np
import torch

# Load data and combine
fcp = FullContextProcessor(data_fpath="../Data/ABC-News/abcnews-sgns-processed.txt", sep="\t")
fcp.appendDf(data_fpath="../Data/Times-of-India/india-sgns-processed.txt", sep="\t")

# Filter to c1-c2 pairs that have occured at least [occurance_thresh] times
cpair_counts = fcp.df.loc[fcp.df["pos"]==1, :].groupby("c1-c2")["pos"].count()
valid_cpairs = cpair_counts[cpair_counts >= 200]
fcp.df = fcp.df.loc[fcp.df["c1-c2"].isin(valid_cpairs.keys()), :]

# Create mappings and save indexed version
fcp.createTwoWayMap(colname="c1-c2")
fcp.createTwoWayMap(colname="word")
fcp.convertColToIdx(colname="c1-c2")
fcp.convertColToIdx(colname="word")

# Turn data in X and y numpy arrays
X = fcp.df.loc[:, ["c1-c2", "word"]].values
y = fcp.df.loc[:, "pos"].values

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
sgns = SGNSClassifier(c_vocab_len = len(fcp.df["c1-c2"].unique()), 
                      w_vocab_len = len(fcp.df["word"].unique()),
                      batch_size = 32,
                      train_epocs = 10,
                      shuffle = True,
                      torch_threads = 7,
                      BCE_reduction = "mean",
                      pred_thresh = 0.5,
                      log_fpath = "./logs/sgns-20-lr-cv.log")

scoring = {"Log-Loss": make_scorer(log_loss), "F1": make_scorer(f1_score)}
param_grid = {"embedding_dim":[5, 20, 50], "lr":[2.0, 1.0, 0.75, 0.5, 0.25]}
gs = GridSearchCV(estimator=sgns,
                  param_grid=param_grid,
                  scoring=scoring,
                  n_jobs=1,
                  cv=[(train_idxs, test_idxs)],
                  refit="Log-Loss",
                  verbose=30)
gs.fit(X, y)

# # Best model is automatically retrained, now get test performance
y_pred = gs.predict(X_test)
print("test logloss: {} | test F1: {}".format(log_loss(y_pred, y_test), f1_score(y_pred, y_test)))

# Save best estimator
best_model = gs.best_estimator_
torch.save(best_model.model_.state_dict(), "sgns-best.pt")

# # load code
# test = SGNSModel(best_model.embedding_dim, best_model.c_vocab_len, best_model.w_vocab_len)
# test.load_state_dict(torch.load("sgns-best-model.pt"))
