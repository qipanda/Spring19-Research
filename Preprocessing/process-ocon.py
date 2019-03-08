from FullContextProcessor import FullContextProcessor

fcp = FullContextProcessor("../Data/OConnor2013/v7.pathfil.dthresh=500.pthresh=10.txt", "\t")
appropriate_pos = {"verb", "noun"}
fnc = lambda x: ",".join(
    [seg.split(",")[1].strip("\"") for seg in x.strip("[]").split("],[") \
        if seg.split(",")[2].strip("\"") in appropriate_pos])

fcp.df.loc[:, "DEP_PATH"] = fcp.df["DEP_PATH"].apply(fnc)
fcp.stackSepCol(colname="DEP_PATH", sep=",", newcolname="WORD")
fcp.writeDf("../Data/OConnor2013/ocon-verb-extracted.txt", "\t")

