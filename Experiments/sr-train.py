import sys, os
# Adding Research directory to path
sys.path.append(os.path.dirname(sys.path[0]))

# Import personal modules
from Models.SGNS import SourceReceiverClassifier, SourceReceiverModel
from Preprocessing.FullContextProcessor import FullContextProcessor

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

sr_class = SourceReceiverClassifier(s_cnt=len(fcp.df["SOURCE"].unique()),
                                    r_cnt=len(fcp.df["RECEIVER"].unique()),
                                    w_cnt=len(fcp.df["WORD"].unique()),
                                    s_std=0.0001,
                                    r_std=0.0001,
                                    w_std=0.1,
                                    K=50,
                                    lr=5e-1,
                                    weight_decay=1e-6,
                                    batch_size = 32,
                                    train_epocs = 1,
                                    log_fpath = "./logs/sr-train-20neg-srstd0.001-wstd0.001.log")
sr_class.fit(X, y)

# Save best estimator
torch.save(sr_class.model_.state_dict(), "sr-best-20neg-srstd0.001-wstd0.001.pt")
