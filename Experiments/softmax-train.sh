#!/usr/bin/env bash
source ENV/bin/activate
python softmax-train.py -g month -K 300 -lr 1.0 -alpha 5e-3 -lam 0.0 -bs 32 -te 50 -lfpth runs
