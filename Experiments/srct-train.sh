#!/usr/bin/env bash
source ENV/bin/activate
python srct-train.py -ns 20 -K 200 -lr 1.0 -alpha 1e-4 -lam 0.0 -bs 32 -train_epochs 5 -lfpth runs
