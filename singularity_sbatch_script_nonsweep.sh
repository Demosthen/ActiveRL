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
#SBATCH --cpus-per-task=6
#
##SBATCH --qos=savio_lowprio
##SBATCH --qos=v100_gpu3_normal
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
#SBATCH --gres=gpu:GTX2080TI:3
##SBATCH --gpus-per-task=1
#
# Wall clock limit (8hrs):
#SBATCH --time=15:59:59
#
# Run 48 examples concurrently
#SBATCH --array=0
BASE_DIR=/global/scratch/users/$USER
LDIR=$BASE_DIR/.local$SLURM_ARRAY_TASK_ID
LOGDIR_BASE=$BASE_DIR/logs
rm -rf $LDIR
mkdir -p $LDIR
SINGULARITY_IMAGE_LOCATION=/global/scratch/users/$USER
SINGULARITY_CACHEDIR=$BASE_DIR/.singularity/cache
export SINGULARITY_CACHEDIR=$BASE_DIR/.singularity/cache
SINGULARITY_TEMPDIR=$BASE_DIR/tmp
export SINGULARITY_TEMPDIR=$BASE_DIR/tmp
SINGULARITY_CACHE_DIR=/global/scratch/users/$USER/transactive-control-social-game
PYTHON_DIR=/global/home/users/$USER/.conda/envs/ActiveRL/bin
export WANDB_API_KEY=87928bf7ce62528545fe624701ab2f3aa25a7547
if test -f sinergym.sif; then
  echo “docker image exists”
else
  singularity build sinergym.sif docker://alejandrocn7/sinergym:latest
fi
singularity run --nv --workdir ./tmp --bind $(pwd):$HOME --bind "$LDIR:$HOME/.local" --bind "$PYTHON_DIR:/.env" sinergym.sif sh -c ". ./singularity_preamble_new.sh && eval $1"









