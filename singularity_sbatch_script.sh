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
#SBATCH --cpus-per-task=4
#
##SBATCH --qos=savio_lowprio
##SBATCH --qos=v100_gpu3_normal
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
#SBATCH --gres=gpu:GTX2080TI:2
##SBATCH --gpus-per-task=1
#
# Wall clock limit (8hrs):
#SBATCH --time=6:58:59
#
# Run 48 examples concurrently
#SBATCH --array=0
BASE_DIR=/global/scratch/users/$USER
LDIR=$BASE_DIR/.local$$
LOGDIR_BASE=$BASE_DIR/logs
rm -rf $LDIR
mkdir -p $LDIR
SINGULARITY_IMAGE_LOCATION=/global/scratch/users/$USER
SINGULARITY_CACHEDIR=$BASE_DIR/.singularity/cache
export SINGULARITY_CACHEDIR=$BASE_DIR/.singularity/cache
SINGULARITY_TEMPDIR=$BASE_DIR/tmp
export SINGULARITY_TEMPDIR=$BASE_DIR/tmp
SINGULARITY_CACHEDIR=/global/scratch/users/$USER/transactive-control-social-game
PYTHON_DIR=/global/home/users/$USER/.conda/envs/ActiveRL/bin
PYTHON_PATH=/home/miniconda/envs/ActiveRL/bin/python
module load gcc
export LD_LIBRARY_PATH=/global/software/sl-7.x86_64/modules/langs/gcc/12.1.0/lib64:${LD_LIBRARY_PATH}
export WANDB_API_KEY=87928bf7ce62528545fe624701ab2f3aa25a7547
WANDB_PATH=/home/miniconda/envs/ActiveRL/bin/wandb
if test -f sinergym.sif; then
  echo “docker image exists”
else
  singularity pull --tmpdir=/global/scratch/users/djang/tmp sinergym.sif docker://doseokjang/sinergym:savio
  #singularity pull sinergym.sif docker://alejandrocn7/sinergym:latest
  #singularity build --tmpdir=$SINGULARITY_TEMPDIR sinergym.sif docker://alejandrocn7/sinergym:latest
fi
#singularity run sinergym.sif sh -c  "ls /home && pwd && ls /usr/bin"
# singularity exec docker://ubuntu:latest cat /etc/issue
# singularity exec sinergym_savio.sif cat /etc/issue
singularity run --nv --workdir ./tmp --bind $(pwd):$HOME --bind "$LDIR:$HOME/.local" --bind "$PYTHON_DIR:/.env" sinergym.sif bash -c ". ./singularity_preamble_new.sh && $PYTHON_PATH -m wandb agent social-game-rl/active-rl/$1 --count 1"
