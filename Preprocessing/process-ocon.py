from FullContextProcessor import FullContextProcessor

fcp = FullContextProcessor("../Data/OConnor2013/v7.pathfil.dthresh=500.pthresh=10.txt", "\t")
appropriate_pos = {"verb", "noun"}
fnc = lambda x: ",".join(
    [seg.split(",")[1].strip("\"") for seg in x.strip("[]").split("],[") \
        if seg.split(",")[2].strip("\"") in appropriate_pos])

fcp.df = fcp.df.iloc[:10]
fcp.df.loc[:, "DEP_PATH"] = fcp.df["DEP_PATH"].apply(fnc)
fcp.stackSepCol(colname="DEP_PATH", sep=",", newcolname="WORD")
fcp.writeDf("../Data/OConnor2013/ocon-verb-extracted.txt", "\t")

import ipdb; ipdb.set_trace()
fcp = FullContextProcessor("../Data/OConnor2013/ocon-verb-extracted.txt", "\t")

# fcp.df.loc[:, "pos"] = 1
# fcp.stackSepCol(colname="ctxt", sep=",", newcolname="word")
# fcp.removeByNanCol("word")
# fcp.removeByNumericCol("word")
# fcp.combineCols("c1", "c2", "-")
# fcp.subsample(t=1e-5, subcolname="word")
# fcp.generateNegSamples(k=10, alpha=0.75, colname="word", negcolname="pos")
# fcp.writeDf("../Data/ABC-News/abcnews-sgns-processed.txt", "\t")

# # Create mappings and save indexed version
# fcp.createTwoWayMap(colname="c1-c2")
# fcp.convertColToIdx(colname="c1-c2")
# fcp.createTwoWayMap(colname="word")
# fcp.convertColToIdx(colname="word")
# fcp.writeDf(fpath="../Data/ABC-News/abcnews-sgns-processed-idx.txt", sep="\t")
