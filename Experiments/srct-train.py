# Adding Research directory to path and parse script arguments
import sys, os, argparse
sys.path.append(os.path.dirname(sys.path[0]))
parser = argparse.ArgumentParser()
parser.add_argument("-ns", "--negsamples", dest="negsamples", type=int,
                    help="number of neg samples per training sample")
parser.add_argument("-K", "--hiddendim", dest="K", type=int,
                    help="hidden dim of pred embeddings, /2 for source and receivers")
parser.add_argument("-lr", "--learnrate", dest="lr", type=float,
                    help="learning rate of SGD")
parser.add_argument("-alpha", dest="alpha", type=float,
                    help="L2 reg scaling term for time diff between sr embeds")
parser.add_argument("-lam", dest="lam", type=float,
                    help="L2 reg scaling term for all embeddings")
parser.add_argument("-bs", "--batchsize", dest="batch_size", type=int,
                    help="batch size for SGD training")
parser.add_argument("-te", "--trainepochs", dest="train_epochs", type=int,
                    help="number of epochs to train via SGD")
parser.add_argument("-lfpth", "-logfpath", dest="log_fpath", type=str,
                    help="where to store tensorboard logging")
args = parser.parse_args()

# Import personal modules
from Models.models import SRCTClassifier, SRCTModel
from Preprocessing.FullContextProcessor import FullContextProcessor

# Import scientific computing modules
import torch
import numpy as np

# Load the data
fcp = FullContextProcessor("../Data/OConnor2013/ocon-nicepaths-indexed.txt", "\t")
X = fcp.df.loc[:, ["SOURCE_IDX", "RECEIVER_IDX", "PRED_IDX", "TIME"]].values
y = np.ones(X.shape[0])

# Negative sample
neg_pred_idxs = fcp.returnNumpyNegSamples(k=args.negsamples,
                                          alpha=0.75,
                                          colname="PRED_IDX",
                                          seed=0)
X_neg = np.tile(X, (args.negsamples, 1))
X_neg[:, 2] = neg_pred_idxs
y_neg = np.zeros(X_neg.shape[0])
X = np.concatenate((X, X_neg), axis=0)
y = np.concatenate((y, y_neg), axis=0)

# #TODO GET RID OF TEST
# X = X[:1000, :]
# y = y[:1000]
# y[-900:] = 0.0

srct_class = SRCTClassifier(s_cnt=len(fcp.df["SOURCE"].unique()),
                            r_cnt=len(fcp.df["RECEIVER"].unique()),
                            p_cnt=len(fcp.df["PRED"].unique()),
                            T=len(fcp.df["TIME"].unique()),
                            K=args.K,
                            lr=args.lr,
                            alpha=args.alpha,
                            lam=args.lam,
                            batch_size=args.batch_size,
                            train_epochs=args.train_epochs,
                            log_fpath=args.log_fpath)
srct_class.fit(X, y)

# Save best estimator
torch.save(srct_class.model_.state_dict(), srct_class.tensorboard_path + ".pt")
