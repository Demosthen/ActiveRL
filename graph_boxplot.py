# %%
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import wandb
import os
import pickle
import time
import writeup.utils as utils
from importlib import reload
from writeup.run_queries import NAMES
run = wandb.init()

runs = [64, 65, 67, 66, 68, 72]# [59, 56, 60, 57, 58, 61]
all_rews = {}
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
    all_rews.update(rews)

    # %%
    print(rews.keys())

    # %%
    
    # if ARTIFACT_TYPE == "scatter":
    #     utils.plot_scatter(time.perf_counter(), rews, bad_idxs, "median_activerl")
    # else:
    #     utils.plot_bars(time.perf_counter(), rews, bad_idxs, "results", only_avg=True)

# %%


# %%
plt.figure(figsize=(6, 4))
fontsize=14
rew_diffs = []
labels = []
for i, key in enumerate(all_rews.keys()):
    if key == 'median_activerl':
        continue
    rew_diff = 100*(all_rews['median_activerl'] - all_rews[key]) / np.abs(all_rews[key])
    name = NAMES["sim2real"][key]
    if name == "Domain Randomization":
        name = "Domain\nRandomization"
    print(f"{key}: {rew_diff.mean()}")
    median = rew_diff.median()["rew"].item()
    
    plt.text((i+.15), (median+.001), str(round(median, 3)), fontsize = fontsize-4)

    rew_diffs.append(rew_diff["rew"])
    labels.append(name)



plt.locator_params(axis='y', nbins=5)
plt.violinplot(rew_diffs, showmeans=True)

plt.xlabel("Baseline", fontsize=fontsize)
plt.xticks(np.arange(1, len(labels)+1), labels)
plt.ylabel("ActiveRL Reward Improvement\nover Baseline (%)", fontsize=fontsize)
ax = plt.gca()
ax.spines[['right', 'top']].set_visible(False)

folder = "writeup/figs/comparisons/" 
os.makedirs(folder, exist_ok=True)
filename = folder + f"violin.png"
plt.tight_layout()
plt.savefig(filename)

wandb.log({"viz": wandb.Image(filename)})
    


