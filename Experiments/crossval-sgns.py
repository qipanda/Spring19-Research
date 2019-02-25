import sys, os
# Adding Research directory to path
sys.path.append(os.path.dirname(sys.path[0]))

# Import personal modules
from Models.SGNS import SGNSClassifier
from Preprocessing.FullContextProcessor import FullContextProcessor

# Import installed modules
import pandas as pd
from sklearn.model_selection import GridSearchCV

# Load data from pickled processor and create two-way maps
proc = FullContextProcessor("../Data/ABC-News/abcnews-sgns-processed.txt", sep="\t")
proc.createTwoWayMap(colname="c1-c2")
proc.createTwoWayMap(colname="word")

# Turn data in X and y numpy arrays
import ipdb; ipdb.set_trace()
