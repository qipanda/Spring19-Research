import sys, os
# Adding Research directory to path
sys.path.append(os.path.dirname(sys.path[0]))

# Import personal modules
from Models.SGNS import SGNSClassifier, SGNSModel
from Preprocessing.FullContextProcessor import FullContextProcessor

# Import installed modules
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

# Set up training for 50 hidden dim model
sgns = SGNSClassifier(embedding_dim = 50,
                      c_vocab_len = len(fcp.df["c1-c2"].unique()), 
                      w_vocab_len = len(fcp.df["word"].unique()),
                      lr = 1.0,
                      batch_size = 32,
                      train_epocs = 10,
                      shuffle = True,
                      torch_threads = 7,
                      BCE_reduction = "mean",
                      pred_thresh = 0.5,
                      log_fpath="./logs/sgns-50-train.log")
sgns.fit(X, y)

# Save model for later
torch.save(sgns.model_.state_dict(), "sgns-50.pt")

# # Load model
# model = SGNSModel(embedding_dim=20,
#                    c_vocab_len = len(fcp.df["c1-c2"].unique()), 
#                    w_vocab_len = len(fcp.df["word"].unique()),)
# model.load_state_dict(torch.load("sgns-20.pt"))
