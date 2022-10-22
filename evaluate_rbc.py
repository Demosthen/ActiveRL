# %%
from citylearn_wrapper import CityLearnEnvWrapper
import multiprocessing as mp
import numpy as np
import pandas as pd
import os
from rbc_agent import RBCAgent
from constants import *

# %%
paths = [f"data/two_buildings/Test_{env_name}" for env_name in CL_ENV_KEYS]
schemas = [os.path.join(cl_path, "schema.json") for cl_path in paths]
env_configs = [{
    "schema": schema,
    "is_evaluation": True,
} for schema in schemas]

# %%
envs = [CityLearnEnvWrapper(env_config) for env_config in env_configs]
def pool_fn(input):
    i, env = input
    obs = env.reset()
    done = False
    obss = []
    next_obss = []
    actions = []
    rews = []
    dones = []
    rbc_agent = RBCAgent(env.action_space)
    while not done:
        action = rbc_agent.compute_action(obs)
        next_obs, rew, done, info = env.step(action)
        actions.append(action)
        rews.append(rew)
        obs = next_obs
    return rews, i

# %%
def store_data(data, path):
    df = pd.Series(data)
    print(df.sum())
    df.to_csv(os.path.join(path, "rbc_rews.csv"), mode='a', index=False, header=False)

def collect_dataset(envs, num_processes=4):
    obss = []
    next_obss = []
    actions = []
    rews = []
    dones = []
    
    pool = mp.Pool(processes=num_processes)
    results = pool.map(pool_fn, enumerate(envs))
    for result, i in results:
        print(i)
        store_data(result, paths[i])

# %%
collect_dataset(envs, 4)

# %%



