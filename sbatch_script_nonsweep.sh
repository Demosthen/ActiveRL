#!/bin/bash
# Job name:
#SBATCH --job-name=socialgame_train
#
# Account:
#SBATCH --account=fc_ntugame
#
# Partition:
#SBATCH --partition=savio3_gpu
#case 
# Number of nodes:
#SBATCH --nodes=1
#
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks=1
#
# Processors per task (please always specify the total number of processors twice the number of GPUs):
#SBATCH --cpus-per-task=8
#
#SBATCH --qos=savio_lowprio
## SBATCH --qos=v100_gpu3_normal
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
#SBATCH --gres=gpu:TITAN:2
##SBATCH --gpus-per-task=1
#
# Wall clock limit (8hrs):
#SBATCH --time=15:59:59
#
# Run 48 examples concurrently
#SBATCH --array=0

export WANDB_API_KEY=87928bf7ce62528545fe624701ab2f3aa25a7547
export PYTHON_PATH=/global/home/users/$USER/.conda/envs/ActiveRL/bin/python
export WANDB_CACHE_DIR=/global/scratch/users/$USER/.cache/wandb
# export LD_LIBRARY_PATH=/global/home/users/lucas_spangher/.conda/pkgs:$LD_LIBRARY_PATH
module load gcc/8.3.0
export MUJOCO_GL=egl
export PATH=$PYTHON_PATH:$PATH
export BCVTB_PATH=/global/scratch/users/$USER/ActiveRL/bcvtb
export EPLUS_PATH=/global/scratch/users/$USER/ActiveRL/EnergyPlus-9-5-0


$PYTHON_PATH $1
