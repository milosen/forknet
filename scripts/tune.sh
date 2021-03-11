#!/bin/bash

source ~/conda/etc/profile.d/conda.sh
conda activate forknet_env

forknet train --slice_dir 0 --epochs 200 --checkpoint 10 \
              --batch_size 32 --lr 0.012

forknet train --slice_dir 1 --epochs 200 --checkpoint 10 \
              --batch_size 32 --lr 0.008

forknet train --slice_dir 2 --epochs 200 --checkpoint 10 \
              --batch_size 32 --lr 0.008