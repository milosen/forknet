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
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#
#SBATCH --mail-type=all
#SBATCH --mail-user=%u@cbs.mpg.de
#
#SBATCH --time=24:00:00
#
################# DON NOT CHANGE THINGS HERE UNLESS YOU KNOW WHAT YOU ARE DOING ###############
# This script is a modification to the implementation suggest by gregSchwartz18 here:
# https://github.com/ray-project/ray/issues/826#issuecomment-522116599
redis_password=$(uuidgen)
export redis_password

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST) # Getting the node names
nodes_array=( $nodes )

node_1=${nodes_array[0]}
ip=$(srun --nodes=1 --ntasks=1 -w $node_1 hostname --ip-address) # making redis-address
port=6379
ip_head=$ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "STARTING HEAD at $node_1"
srun --nodes=1 --ntasks=1 -w $node_1 start-head.sh $ip $redis_password &
sleep 30

worker_num=$(($SLURM_JOB_NUM_NODES - 1)) #number of nodes other than the head node
for ((  i=1; i<=$worker_num; i++ ))
do
  node_i=${nodes_array[$i]}
  echo "STARTING WORKER $i at $node_i"
  srun --nodes=1 --ntasks=1 -w $node_i start-worker.sh $ip_head $redis_password &
  sleep 5
done
##############################################################################################
module purge
module load miniconda cuda

source activate forknet

# Run the program:
srun forknet tune --gpus_per_trial=2 --cpus_per_trial=1 --max_epochs=100 \
		  --num_samples=3 --distributed &> rtune_forknet.out
