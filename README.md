# ActiveRL

Instructions for getting this working on savio:

1. Clone repo
2. cd /global/home/users/$USER/
3. mv .conda /global/scratch/users/$USER/.conda
4. ln -s /global/scratch/users/$USER/.conda .conda
5. Navigate to wherever you cloned the repo
6. conda install -f environment.yml
7. conda activate ActiveRL
8. pip install git+https://github.com/cooper-org/cooper.git
9. pip install -e gym-simplegrid/ --no-deps
10. pip install moviepy==2.1.3.dev6
11. python ./download_model.py

To run a wandb sweep:
./master_sbatch_script.sh WANDB_SWEEP_ID NUMBER_OF_RUNS

To run an individual run:
./master_sbatch_script_nonsweep.sh "COMMAND LINE ARGS NOT INCLUDING THE ACTUAL PYTHON COMMAND" NUMBER_OF_RUNS
For example: 
./master_sbatch_script_nonsweep.sh "run_experiments.py --num_timesteps=400000 --gw_steps_per_cell=10 --wandb --env=gw --use_coop --gw_filename=gridworlds/good_bubble.txt --num_descent_steps=10 --seed=1234567 --use_activerl=1" 1
