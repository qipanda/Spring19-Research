import pickle
from FullContextProcessor import FullContextProcessor

fcp = FullContextProcessor("../Data/ABC-News/abcnews-complete-ctxt.txt", "\t")
fcp.df.loc[:, "pos"] = 1
fcp.stackSepCol(colname="ctxt", sep=",", newcolname="word")
fcp.removeByNanCol("word")
fcp.removeByNumericCol("word")
fcp.combineCols("c1", "c2", "-")
fcp.subsample(t=1e-5, subcolname="word")
fcp.generateNegSamples(k=10, alpha=0.75, colname="word", negcolname="pos")
fcp.writeDf("../Data/ABC-News/abcnews-sgns-processed.txt", "\t")

# Create mappings and save indexed version
fcp.createTwoWayMap(colname="c1-c2")
fcp.convertColToIdx(colname="c1-c2")
fcp.createTwoWayMap(colname="word")
fcp.convertColToIdx(colname="word")
fcp.writeDf(fpath="../Data/ABC-News/abcnews-sgns-processed-idx.txt", sep="\t")
