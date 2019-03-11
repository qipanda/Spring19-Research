import sys, os
# Adding Research directory to path
sys.path.append(os.path.dirname(sys.path[0]))

# Import personal modules
from Models.SGNS import SourceReceiverClassifier, SourceReceiverModel
from Preprocessing.FullContextProcessor import FullContextProcessor

# Import installed modules
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import make_scorer, log_loss, f1_score, precision_score, recall_score, accuracy_score
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
sr_class = SourceReceiverClassifier(s_cnt=len(fcp.df["SOURCE"].unique()),
                                    r_cnt=len(fcp.df["RECEIVER"].unique()),
                                    w_cnt=len(fcp.df["WORD"].unique()),
                                    batch_size = 32,
                                    train_epocs = 3,
                                    log_fpath = "./logs/sr-cv-junk.log")

scoring = {
    "Log-Loss": make_scorer(log_loss),
    "Accuracy": make_scorer(accuracy_score),
    "Precision": make_scorer(precision_score),
    "Recall": make_scorer(recall_score),
    "F1": make_scorer(f1_score),
}
param_grid = {
    "K":[50, 100],
    "lr":[1e0, 1e-1],
}

gs = GridSearchCV(estimator=sr_class,
                  param_grid=param_grid,
                  scoring=scoring,
                  n_jobs=1,
                  cv=[(train_idxs, test_idxs)],
                  refit="Log-Loss",
                  verbose=30)
gs.fit(X, y)

# # Best model is automatically retrained, now get test performance
y_pred = gs.predict(X_test)
print("test logloss: {} | Acc: {} | Prec: {} | Rec: {} | test F1: {}".\
    format(log_loss(y_test, y_pred), 
           accuracy_score(y_test, y_pred),
           precision_score(y_test, y_pred), 
           recall_score(y_test, y_pred),
           f1_score(y_test, y_pred)),)

# # Save best estimator
# best_model = gs.best_estimator_
# torch.save(best_model.model_.state_dict(), "sr-best-20neg.pt")

# # load code
# test = SGNSModel(best_model.embedding_dim, best_model.c_vocab_len, best_model.w_vocab_len)
# test.load_state_dict(torch.load("sgns-best-model.pt"))
