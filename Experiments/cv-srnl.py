import sys, os
# Adding Research directory to path
sys.path.append(os.path.dirname(sys.path[0]))

# Import personal modules
from Models.SGNS import SourceReceiverNonlinearClassifier, SourceReceiverNonlinearModel
from Preprocessing.FullContextProcessor import FullContextProcessor

# Import installed modules
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import make_scorer, log_loss, f1_score, precision_score, recall_score
import torch
import numpy as np

fcp = FullContextProcessor("../Data/OConnor2013/ocon-verb-noun-extracted.txt", "\t")

fcp.createTwoWayMap("SOURCE")
fcp.createTwoWayMap("RECEIVER")
fcp.createTwoWayMap("WORD")
fcp.convertColToIdx("SOURCE")
fcp.convertColToIdx("RECEIVER")
fcp.convertColToIdx("WORD")

X = fcp.df.loc[:, ["SOURCE", "RECEIVER", "WORD"]].values
y = np.ones(X.shape[0])

k = 10
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
srnl_class = SourceReceiverNonlinearClassifier( s_cnt=len(fcp.df["SOURCE"].unique()),
                                                r_cnt=len(fcp.df["RECEIVER"].unique()),
                                                w_cnt=len(fcp.df["WORD"].unique()),
                                                batch_size = 32,
                                                train_epocs = 10,
                                                shuffle = True,
                                                torch_threads = 7,
                                                BCE_reduction = "mean",
                                                pred_thresh = 0.5,
                                                log_fpath = "./logs/srnl-cv.log")

scoring = {
    "Log-Loss": make_scorer(log_loss),
    "Precision": make_scorer(precision_score),
    "Recall": make_scorer(recall_score),
    "F1": make_scorer(f1_score),
}
param_grid = {"K":[5, 20, 50], "lr":[1.0, 0.1, 0.01], "alpha":[2.0, 1.0, 0.1]}
gs = GridSearchCV(estimator=srnl_class,
                  param_grid=param_grid,
                  scoring=scoring,
                  n_jobs=1,
                  cv=[(train_idxs, test_idxs)],
                  refit="Log-Loss",
                  verbose=30)
gs.fit(X, y)

# # Best model is automatically retrained, now get test performance
y_pred = gs.predict(X_test)
print("test logloss: {} | Precision: {} | Recall: {} | test F1: {}".\
    format(log_loss(y_pred, y_test), 
           precision_score(y_pred, y_test), 
           recall_score(y_pred, y_test),
           f1_score(y_pred, y_test)))

# Save best estimator
best_model = gs.best_estimator_
torch.save(best_model.model_.state_dict(), "srnl-best.pt")

# # load code
# test = SGNSModel(best_model.embedding_dim, best_model.c_vocab_len, best_model.w_vocab_len)
# test.load_state_dict(torch.load("sgns-best-model.pt"))