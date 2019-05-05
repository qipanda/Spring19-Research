#!/usr/bin/env bash
source ../ENV/bin/activate
python softmax-train.py -g month -t 251 -K 300 -lr 1.0 -alpha 1e-2 -lam 0.0 -bs 32 -te 50 -lfpth runs
