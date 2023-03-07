#!/bin/bash
export PATH=/home/miniconda/envs/ActiveRL/bin/:/global/home/users/$USER/.conda/envs/ActiveRL/bin:/global/home/users/$USER/.local/bin:$PATH

PYTHON_PATH=/home/miniconda/envs/ActiveRL/bin/python
alias python=$PYTHON_PATH
flock -x /global/scratch/users/djang/ActiveRL/singularity_package_lock -c "$PYTHON_PATH -m pip install -e gridworld/gym_simplegrid --no-deps"

