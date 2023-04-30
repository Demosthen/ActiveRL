# %%
import wandb
import os
import pickle
import time
import writeup.utils as utils
from importlib import reload
run = wandb.init()
runs = [59, 56, 60, 57, 58, 61]
for run_id in runs:
    artifact = run.use_artifact(f'doseok/ActiveRL/reward_data:v{run_id}', type='evaluation')
    artifact_dir = artifact.download()
    ARTIFACT_TYPE = "scatter"

    # %%
    
    file_name = list(os.listdir(artifact_dir))[0]
    file_name = os.path.join(artifact_dir, file_name)
    print(file_name)
    with open(file_name, 'rb') as f:
        data = pickle.load(f)
    rews, bad_idxs = data

    # %%
    print(rews.keys())

    # %%
    
    if ARTIFACT_TYPE == "scatter":
        utils.plot_scatter(time.perf_counter(), rews, bad_idxs, "median_activerl")
    else:
        utils.plot_bars(time.perf_counter(), rews, bad_idxs, "results", only_avg=True)

# %%


# %%


