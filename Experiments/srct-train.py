import sys, os
# Adding Research directory to path
sys.path.append(os.path.dirname(sys.path[0]))

# Import personal modules
from Models.SGNS import SRCTClassifier, SRCTModel
from Preprocessing.FullContextProcessor import FullContextProcessor

import torch
import numpy as np

# fcp = FullContextProcessor("../Data/OConnor2013/ocon-verb-noun-extracted.txt", "\t")
fcp = FullContextProcessor("../Data/OConnor2013/ocon-nicepaths-extracted.txt", "\t")

fcp.createTwoWayMap("SOURCE")
fcp.createTwoWayMap("RECEIVER")
fcp.createTwoWayMap("PRED")
fcp.convertColToIdx("SOURCE")
fcp.convertColToIdx("RECEIVER")
fcp.convertColToIdx("PRED")
fcp.createMonthTimeIdx("DATE", "TIME")

X = fcp.df.loc[:, ["SOURCE", "RECEIVER", "PRED", "TIME"]].values
y = np.ones(X.shape[0])

k = 20
neg_pred_idxs = fcp.returnNumpyNegSamples(k=k,
                                          alpha=0.75,
                                          colname="PRED",
                                          seed=0)
X_neg = np.tile(X, (k, 1))
X_neg[:, 2] = neg_pred_idxs
y_neg = np.zeros(X_neg.shape[0])

X = np.concatenate((X, X_neg), axis=0)
y = np.concatenate((y, y_neg), axis=0)

# #TODO GET RID OF TEST
# X = X[:100, :]
# y = y[:100]
# y[-90:] = 0.0

srct_class = SRCTClassifier(s_cnt=len(fcp.df["SOURCE"].unique()),
                            r_cnt=len(fcp.df["RECEIVER"].unique()),
                            p_cnt=len(fcp.df["PRED"].unique()),
                            T=len(fcp.df["TIME"].unique()),
                            K=200,
                            lr=1.0,
                            alpha=10e0,
                            lam=1e-9,
                            batch_size = 32,
                            train_epocs = 1,
                            log_fpath = "./logs/srct-train.log")
srct_class.fit(X, y)

# Save best estimator
torch.save(srct_class.model_.state_dict(), "srct-K200-lr1-alpha10e1-lam1e-9.pt")
