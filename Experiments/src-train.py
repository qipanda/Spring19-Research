import sys, os
# Adding Research directory to path
sys.path.append(os.path.dirname(sys.path[0]))

# Import personal modules
from Models.SGNS import SourceReceiverConcatClassifier, SourceReceiverConcatModel
from Preprocessing.FullContextProcessor import FullContextProcessor

import torch
import numpy as np

# fcp = FullContextProcessor("../Data/OConnor2013/ocon-verb-noun-extracted.txt", "\t")
fcp = FullContextProcessor("../Data/OConnor2013/ocon-nicepaths-extracted.txt", "\t")

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

src_class = SourceReceiverConcatClassifier(s_cnt=len(fcp.df["SOURCE"].unique()),
                                    r_cnt=len(fcp.df["RECEIVER"].unique()),
                                    w_cnt=len(fcp.df["WORD"].unique()),
                                    xavier=True,
                                    K_s=100,
                                    K_r=100,
                                    K_w=200,
                                    lr=5e-1,
                                    # weight_decay=1e-6,
                                    batch_size = 32,
                                    train_epocs = 1,
                                    log_fpath = "./logs/src-train-xavier.log")
src_class.fit(X, y)

# Save best estimator
torch.save(src_class.model_.state_dict(), "src-xavier.pt")
