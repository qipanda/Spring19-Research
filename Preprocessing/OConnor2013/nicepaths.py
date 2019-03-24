import json, sys
import util

with open("../../Data/OConnor2013/v7.pathfil.dthresh=500.pthresh=10.txt", "r") as fr,\
     open("../../Data/OConnor2013/ocon-nicepaths-extracted.txt", "w") as fw:
    lines = fr.readlines()
    fw.write(lines[0] + "\n")
    for line in lines[1:]:
        parts = line.rstrip('\n').split('\t')
        parts[-1] = util.nicepath(parts[-1], html=False)
        fw.write('\t'.join(parts) + "\n")
