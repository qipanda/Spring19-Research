import sys, os
# Adding Research directory to path
sys.path.append(os.path.dirname(sys.path[0]))

# Import personal modules
from Models.SGNS import SGNSClassifier
from Preprocessing.FullContextProcessor import FullContextProcessor

# Import installed modules
import pandas as pd
from sklearn.model_selection import GridSearchCV

# Load data from indexed version
fcp = FullContextProcessor("../Data/ABC-News/abcnews-sgns-processed-idx.txt", sep="\t")

# Turn data in X and y numpy arrays
X = fcp.df.loc[:, ["c1-c2", "word"]].values
y = fcp.df.loc[:, "pos"].values

# Set up cross validation
