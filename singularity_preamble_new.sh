#!/bin/bash
export PATH=/home/miniconda/envs/ActiveRL/bin/:/global/home/users/$USER/.conda/envs/ActiveRL/bin:/global/home/users/$USER/.local/bin:$PATH

PYTHON_PATH=/home/miniconda/envs/ActiveRL/bin/python
alias python=$PYTHON_PATH
flock -x /global/scratch/users/djang/ActiveRL/singularity_package_lock -c "$PYTHON_PATH -m pip install -e gym-simplegrid \
&& $PYTHON_PATH -m pip install dm_control==1.0.9 \
&& $PYTHON_PATH -m pip install dm2gym==0.2.0 \
&& $PYTHON_PATH -m pip install gym==0.24.1"

