#!/bin/bash -l
# Standard output and error:
#SBATCH -o /u/nmilosevic/forknet/forknet_gpu_job.out.%j
#SBATCH -e /u/nmilosevic/forknet/forknet_gpu_job.err.%j
#
# Initial working directory:
#SBATCH -D /u/nmilosevic/forknet
#
#SBATCH -J forknet
#
#SBATCH --partition="gpu"
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:v100:2         # If using both GPUs of a node
#SBATCH --mem=92500
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#
#SBATCH --mail-type=all
#SBATCH --mail-user=%u@cbs.mpg.de
#
#SBATCH --time=24:00:00
module purge
module load miniconda cuda

source activate forknet_env 

# Run the program:
srun scripts/train-net.sh &> train_forknet.out
