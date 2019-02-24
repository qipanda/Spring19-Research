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

# Pickle object to keep twoway maps and save dataframe as txt
fcp.writeDf("../Data/ABC-News/abcnews-sgns-processed.txt", "\t")
with open("../Data/ABC-News/abcnews-sgns-processed.pickle", "wb") as handle:
    pickle.dump(fcp, handle)
