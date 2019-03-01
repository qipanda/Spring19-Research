import pickle
from FullContextProcessor import FullContextProcessor

"""
Script for taking abcnews data with complete context window and distilling it
down for sgns processing
"""

fcp = FullContextProcessor("../Data/Times-of-India/india-complete-ctxt.txt", "\t")
fcp.df.loc[:, "pos"] = 1
print("stacking...")
fcp.stackSepCol(colname="ctxt", sep=",", newcolname="word")
fcp.removeByNanCol("word")
fcp.removeByNumericCol("word")
fcp.combineCols("c1", "c2", "-")
print("subsampling...")
fcp.subsample(t=1e-5, subcolname="word")
print("generating negsamples...")
fcp.generateNegSamples(k=10, alpha=0.75, colname="word", negcolname="pos")
fcp.writeDf("../Data/Times-of-India/india-sgns-processed.txt", "\t")

# Create mappings and save indexed version
fcp.createTwoWayMap(colname="c1-c2")
fcp.convertColToIdx(colname="c1-c2")
fcp.createTwoWayMap(colname="word")
fcp.convertColToIdx(colname="word")
fcp.writeDf(fpath="../Data/Times-of-India/india-sgns-processed-idx.txt", sep="\t")
