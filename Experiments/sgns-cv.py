import pickle
import sys, os
# Adding Research directory to path
sys.path.append(os.path.dirname(sys.path[0]))

# Import personal modules
from Models.SGNS import SGNSClassifier
from Preprocessing.FullContextProcessor import FullContextProcessor

# Import installed modules
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, log_loss, f1_score

# Load data from indexed version
fcp = FullContextProcessor("../Data/ABC-News/abcnews-sgns-processed-idx.txt", sep="\t")

# Turn data in X and y numpy arrays
X = fcp.df.loc[:, ["c1-c2", "word"]].values
y = fcp.df.loc[:, "pos"].values

# Set up cross validation, want to evaluate on BCE
sgns = SGNSClassifier(c_vocab_len = len(fcp.df["c1-c2"].unique()), 
                      w_vocab_len = len(fcp.df["word"].unique()),
                      lr = 1e-3,
                      train_epocs = 20,
                      torch_threads = 5,
                      BCE_reduction = "mean",
                      pred_thresh = 0.5,)

scoring = {"Log-Loss": make_scorer(log_loss), "F1": make_scorer(f1_score)}
param_grid = {"embedding_dim":[5, 20, 50, 100, 300]}
gs = GridSearchCV(estimator=sgns,
                  param_grid=param_grid,
                  scoring=scoring,
                  n_jobs=10,
                  cv=5,
                  refit="Log-Loss",
                  verbose=30)

gs.fit(X, y)
results = gs.cv_results_

# Save cv results
with open("sgns_cv_results.pickle", "wb") as handle:
    pickle.dump(results, handle)

with open("sgns_cv_results.pickle", "rb") as handle:
    results = pickle.load(handle)
