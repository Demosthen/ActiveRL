# ActiveRL

This is a repository containing the code that was used to run the experiments for the paper [Active Reinforcement Learning for Robust Building Control](https://arxiv.org/abs/2312.10289).

## Setup

### Instructions for general machines:
1. Clone repo
5. Navigate to wherever you cloned the repo
6. conda env create -f environment.yml
7. conda activate ActiveRL
8. pip install git+https://github.com/cooper-org/cooper.git
9. pip install -e gridworld/gym-simplegrid/ --no-deps
10. pip install moviepy==1.0.3
11. pip uninstall pygame
12. Install bcvtb and EnergyPlus like in step 2 and 3 of https://github.com/ugr-sail/sinergym
12. pip install sinergym[extras]
13. pip install gym==0.24.1
14. pip install -e gym-simplegrid
15. pip install dm_control==1.0.9

### Instructions for the Savio compute cluster:

1. Clone repo
2. cd /global/home/users/$USER/
3. mv .conda /global/scratch/users/$USER/.conda
4. ln -s /global/scratch/users/$USER/.conda .conda
5. Navigate to wherever you cloned the repo
6. conda install -f environment.yml
7. conda activate ActiveRL
8. pip install git+https://github.com/cooper-org/cooper.git
9. pip install -e gym-simplegrid/ --no-deps
10. pip install moviepy==1.0.3
11. pip uninstall pygame
12. pip install sinergym[extras]
13. pip install gym==0.24.1

## Running with SLURM
To run a wandb sweep on a cluster using SLURM:
./master_sbatch_script.sh WANDB_SWEEP_ID NUMBER_OF_RUNS

To run an individual run on a cluster using SLURM:
./master_sbatch_script_nonsweep.sh "COMMAND LINE ARGS NOT INCLUDING THE ACTUAL PYTHON COMMAND" NUMBER_OF_RUNS
For example: 
./master_sbatch_script_nonsweep.sh "run_experiments.py --num_timesteps=400000 --gw_steps_per_cell=10 --wandb --env=gw --no_coop --gw_filename=gridworlds/good_bubble.txt --num_descent_steps=10 --seed=1234567 --use_activerl=1" 1

If running in a singularity container, use ./master_singularity_sbatch_script.sh and ./master_singularity_sbatch_nonsweep.sh instead


## Commands for Reproducibility
Here are the individual commands that were used to generate data for the paper:

Seeds: 8765, 87654, 876543, 8765432, 87654321

ActiveRL:
"run_experiments.py --num_timesteps=7500000 --wandb --env=sg --num_gpus=1 --train_batch_size=26280 --horizon=4380 --num_training_workers=3 --num_eval_workers=3 --eval_interval=3 --num_envs_per_worker=2 --num_descent_steps=20 --clip_param=0.3 --gamma=0.8 --lr=5e-05 --num_sgd_iter=40 --dropout=0.1 --continuous --activerl_lr=0.01 --activerl_reg_coeff=0.5 --dropout=0.1 --num_descent_steps=91 --num_dropout_evals=10 --only_vary_offset --seed=SEED --sinergym_sweep=1.0,0,0,0" 

RBC:
"run_experiments.py --num_timesteps=7500000 --wandb --env=sg --num_gpus=1 --train_batch_size=26280 --horizon=4380 --num_training_workers=3 --num_eval_workers=3 --eval_interval=3 --num_envs_per_worker=2 --num_descent_steps=20 --clip_param=0.3 --gamma=0.8 --lr=5e-05 --num_sgd_iter=40 --dropout=0.1 --continuous --activerl_lr=0.01 --activerl_reg_coeff=0.5 --dropout=0.1 --num_descent_steps=91 --num_dropout_evals=10 --only_vary_offset --seed=SEED --sinergym_sweep=0,1,0,0"

RL:
"run_experiments.py --num_timesteps=7500000 --wandb --env=sg --num_gpus=1 --train_batch_size=26280 --horizon=4380 --num_training_workers=3 --num_eval_workers=3 --eval_interval=3 --num_envs_per_worker=2 --num_descent_steps=20 --clip_param=0.3 --gamma=0.8 --lr=5e-05 --num_sgd_iter=40 --dropout=0.1 --continuous --activerl_lr=0.01 --activerl_reg_coeff=0.5 --dropout=0.1 --num_descent_steps=91 --num_dropout_evals=10 --only_vary_offset --seed=SEED --sinergym_sweep=0,0,0,0"

Domain Randomization:
"run_experiments.py --num_timesteps=7500000 --wandb --env=sg --num_gpus=1 --train_batch_size=26280 --horizon=4380 --num_training_workers=3 --num_eval_workers=3 --eval_interval=3 --num_envs_per_worker=2 --num_descent_steps=20 --clip_param=0.3 --gamma=0.8 --lr=5e-05 --num_sgd_iter=40 --dropout=0.1 --continuous --activerl_lr=0.01 --activerl_reg_coeff=0.5 --dropout=0.1 --num_descent_steps=91 --num_dropout_evals=10 --only_vary_offset --seed=SEED --sinergym_sweep=0,0,0,1.0"

ActivePLR:
"run_experiments.py --num_timesteps=7500000 --wandb --env=sg --num_gpus=1 --train_batch_size=26280 --horizon=4380 --num_training_workers=3 --num_eval_workers=3 --eval_interval=3 --num_envs_per_worker=2 --num_descent_steps=20 --clip_param=0.3 --gamma=0.8 --lr=5e-05 --num_sgd_iter=40 --dropout=0.1 --continuous --activerl_lr=0.01 --activerl_reg_coeff=0.5 --dropout=0.1 --num_descent_steps=91 --num_dropout_evals=10 --only_vary_offset --plr_d=1 --seed=SEED --sinergym_sweep=1.0,0,0,0"

Active RPLR:
"run_experiments.py --num_timesteps=7500000 --wandb --env=sg --num_gpus=1 --train_batch_size=26280 --horizon=4380 --num_training_workers=3 --num_eval_workers=3 --eval_interval=3 --num_envs_per_worker=2 --num_descent_steps=20 --clip_param=0.3 --gamma=0.8 --lr=5e-05 --num_sgd_iter=40 --dropout=0.1 --continuous --activerl_lr=0.01 --activerl_reg_coeff=0.5 --dropout=0.1 --num_descent_steps=91 --num_dropout_evals=10 --only_vary_offset --plr_d=1 --plr_robust --seed=SEED --sinergym_sweep=1.0,0,0,0"

RPLR:
"run_experiments.py --num_timesteps=3000000 --wandb --env=sg --num_gpus=1 --train_batch_size=26280 --horizon=4380 --num_training_workers=3 --num_eval_workers=3 --eval_interval=3 --num_envs_per_worker=2 --num_descent_steps=20 --clip_param=0.3 --gamma=0.8 --lr=5e-05 --num_sgd_iter=40 --dropout=0.1 --continuous --activerl_lr=0.01 --activerl_reg_coeff=0.5 --dropout=0.1 --num_descent_steps=91 --num_dropout_evals=10 --only_vary_offset --plr_d=1 --plr_robust --plr_beta=0.03362866617598082 --plr_envs_to_1=50 --plr_rho=0.0064809998847552355 --seed=8765 --sinergym_sweep=0,0,0,1.0"
