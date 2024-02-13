# THIS FILE IS INTENDED TO BE RUN FROM THE ROOT DIRECTORY, NOT THE CITYLEARN_MODEL_TRAINING SUBDIRECTORY
# %%
import pandas as pd

if __name__ == "__main__":
    # For some reason we have to define this store before importing the citylearnwrapper...
    store = pd.HDFStore("citylearn_model_training/planning_model_data.h5")
import sys

# setting path
sys.path.append("./")
from citylearn_wrapper import CityLearnEnvWrapper
import csv

import numpy as np
import h5py
import multiprocessing as mp

# %%
schemas = [
    "data/Test_cold_Texas/schema.json",
    "data/Test_dry_Cali/schema.json",
    "data/Test_hot_new_york/schema.json",
    "data/Test_snowy_Cali_winter/schema.json",
]
env_configs = [
    {
        "schema": schema,
        "is_evaluation": True,
    }
    for schema in schemas
]
num_episodes = 25000

# %%
envs = [CityLearnEnvWrapper(env_config) for env_config in env_configs]


def pool_fn(env):
    obs = env.reset()
    done = False
    obss = []
    next_obss = []
    actions = []
    rews = []
    dones = []
    while not done:
        action = env.action_space.sample()
        next_obs, rew, done, info = env.step(action)
        # data.append([obs, next_obs, action.tolist(), rew, done])
        obss.append(obs)
        next_obss.append(next_obs)
        actions.append(action)
        rews.append([rew])
        dones.append([done])
        obs = next_obs
    return obss, next_obss, actions, rews, dones


# %%
def collect_dataset(envs, num_episodes, num_processes=2):
    obss = []
    next_obss = []
    actions = []
    rews = []
    dones = []
    pool = mp.Pool(processes=num_processes)
    for i in range(0, num_episodes, num_processes):
        print(f"Episode {i}, {str(store.info())}")
        results = pool.map(
            pool_fn, [envs[np.random.choice(len(envs))] for _ in range(num_processes)]
        )
        for result in results:
            store_data(result)
            obss = []
            next_obss = []
            actions = []
            rews = []
            dones = []
    return obss, next_obss, actions, rews, dones


def convert_to_df(data, prefix):
    return pd.DataFrame(data, columns=[f"{prefix}_{i}" for i in range(len(data[0]))])


def store_data(data):
    obss, next_obss, actions, rews, dones = data
    obss_df = convert_to_df(obss, "obs")
    next_obss_df = convert_to_df(next_obss, "next_obs")
    actions_df = convert_to_df(actions, "action")
    rews_df = convert_to_df(rews, "rew")
    dones_df = convert_to_df(dones, "done")
    store.append("obs", obss_df, format="t")
    store.append("next_obs", next_obss_df, format="t")
    store.append("actions", actions_df, format="t")
    store.append("rews", rews_df, format="t")
    store.append("dones", dones_df, format="t")


# %%
if __name__ == "__main__":
    # About 75 trajectories in a gigabyte

    data = collect_dataset(envs, num_episodes, 30)
    store.close()
# store_data(data, "planning_model_data.pkl")

# %%


# %%


# %%
