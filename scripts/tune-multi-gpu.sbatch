#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./forknet_tune_job.out.%j
#SBATCH -e ./forknet_tune_job.err.%j
#
# Initial working directory:
#SBATCH -D .
#
#SBATCH -J forknet
#
#SBATCH --partition="gpu"
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:v100:2         # If using both GPUs of a node
#SBATCH --mem=92500
#
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#
#SBATCH --mail-type=all
#SBATCH --mail-user=nmilosevic@cbs.mpg.de
#
#SBATCH --time=24:00:00
module purge
module load miniconda cuda

source activate forknet

# Run the program:
srun forknet tune --gpus_per_trial=2 --cpus_per_trial=2 --max_epochs=500 \
                  --num_samples=2 --distributed &> rtune_forknet.out
