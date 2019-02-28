import sys, os
# Adding Research directory to path
sys.path.append(os.path.dirname(sys.path[0]))

# Import personal modules
from Models.SGNS import SGNSClassifier, SGNSModel
from Preprocessing.FullContextProcessor import FullContextProcessor

# Import installed modules
import torch

# Load data from indexed version
fcp = FullContextProcessor("../Data/ABC-News/abcnews-sgns-processed-idx.txt", sep="\t")

# Turn data in X and y numpy arrays
X = fcp.df.loc[:, ["c1-c2", "word"]].values
y = fcp.df.loc[:, "pos"].values

# Set up training for 20 hidden dim model
sgns = SGNSClassifier(embedding_dim = 20,
                      c_vocab_len = len(fcp.df["c1-c2"].unique()), 
                      w_vocab_len = len(fcp.df["word"].unique()),
                      lr = 1e-3,
                      train_epocs = 20,
                      torch_threads = 5,
                      BCE_reduction = "mean",
                      pred_thresh = 0.5,)
print("training 20 dim...")
sgns.fit(X, y)

# Save model for later
torch.save(sgns.model_.state_dict(), "sgns-20.pt")

# # Load model
# model = SGNSModel(embedding_dim=2,
#                    c_vocab_len = len(fcp.df["c1-c2"].unique()), 
#                    w_vocab_len = len(fcp.df["word"].unique()),)
# model.load_state_dict(torch.load("sgns-20.pt"))


# Set up training for 300 hidden dim model
sgns = SGNSClassifier(embedding_dim = 300,
                      c_vocab_len = len(fcp.df["c1-c2"].unique()), 
                      w_vocab_len = len(fcp.df["word"].unique()),
                      lr = 1e-3,
                      train_epocs = 20,
                      torch_threads = 5,
                      BCE_reduction = "mean",
                      pred_thresh = 0.5,)
print("training 300 dim...")
sgns.fit(X, y)

# Save model for later
torch.save(sgns.model_.state_dict(), "sgns-300.pt")

# Load model
# model = SGNSModel(embedding_dim=300,
#                    c_vocab_len = len(fcp.df["c1-c2"].unique()), 
#                    w_vocab_len = len(fcp.df["word"].unique()),)
# model.load_state_dict(torch.load("sgns-300.pt"))
