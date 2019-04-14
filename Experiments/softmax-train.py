# Adding Research directory to path and parse script arguments
import sys, os, argparse
sys.path.append(os.path.dirname(sys.path[0]))
parser = argparse.ArgumentParser()
parser.add_argument("-g", "--granularity", dest="gran", type=str,
                    help="the granularity of time step {year, month}")
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
from Models.models import SRCTSoftmaxClassifier, SRCTSoftmaxModel
from Preprocessing.FullContextProcessor import FullContextProcessor

# Import scientific computing modules
import torch
import numpy as np

# Load the data
fcp = FullContextProcessor(
    "../Data/OConnor2013/ocon-nicepaths-{}-indexed.txt".format(args.gran), "\t")
X = fcp.df.loc[:, ["SOURCE_IDX", "RECEIVER_IDX", "TIME"]].values
y = fcp.df.loc[:, "PRED_IDX"].values

softmax_class = SRCTSoftmaxClassifier(s_cnt=len(fcp.df["SOURCE"].unique()),
                                    r_cnt=len(fcp.df["RECEIVER"].unique()),
                                    p_cnt=len(fcp.df["PRED"].unique()),
                                    T=len(fcp.df["TIME"].unique()),
                                    K=args.K,
                                    lr=args.lr,
                                    alpha=args.alpha,
                                    lam=args.lam,
                                    batch_size=args.batch_size,
                                    train_epochs=args.train_epochs,
                                    log_fpath=args.log_fpath + "/" + args.gran)
softmax_class.fit(X, y)

# Save best estimator
torch.save(softmax_class.model_.state_dict(), "{}_".format(args.gran) + softmax_class.tensorboard_path + ".pt")

# TODO Create torch embeddings?
