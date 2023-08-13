# %%
import wandb
import os
import pickle
import time
import writeup.utils as utils
from importlib import reload
run = wandb.init()
# insert the ids of the 
runs = [83]#[81]# [64, 65, 67, 66, 68, 72, 79]
for run_id in runs:
    artifact = run.use_artifact(f'doseok/ActiveRL/reward_data:v{run_id}', type='evaluation')
    artifact_dir = artifact.download()
    ARTIFACT_TYPE = "bar"

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
        utils.plot_bars(time.perf_counter(), rews, bad_idxs, "results", only_avg=False, relative=True)

# %%


# %%



