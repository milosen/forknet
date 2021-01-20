#!/bin/bash -l
# Standard output and error:
#SBATCH -o ./forknet_gpu_job.out.%j
#SBATCH -e ./forknet_gpu_job.err.%j
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
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#
#SBATCH --mail-type=all
#SBATCH --mail-user=%u@cbs.mpg.de
#
#SBATCH --time=24:00:00
module purge
module load miniconda cuda

source activate forknet

# Run the program:
srun forknet train &> train_forknet.out
