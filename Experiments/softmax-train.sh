#!/usr/bin/env bash
source ENV/bin/activate
python softmax-train.py -g month -ns 20 -K 200 -lr 1.0 -alpha 1e-5 -lam 0.0 -bs 32 -te 30 -lfpth runs
python softmax-train.py -g month -ns 20 -K 200 -lr 1.0 -alpha 1e-4 -lam 0.0 -bs 32 -te 30 -lfpth runs
python softmax-train.py -g month -ns 20 -K 200 -lr 1.0 -alpha 1e-3 -lam 0.0 -bs 32 -te 30 -lfpth runs
