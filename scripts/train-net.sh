#!/bin/bash

forknet train-net --checkpoint 10 --batch_size 32 --epochs 100 --lr 0.001 --n_train 260 --n_val 28
