#!/usr/bin/env bash
source ENV/bin/activate
python srct-train.py -g month -ns 20 -K 200 -lr 1.0 -alpha 1e-5 -lam 0.0 -bs 32 -te 10 -lfpth runs
