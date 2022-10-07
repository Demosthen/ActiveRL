# %%
import pandas as pd
# For some reason we have to define this store before importing the citylearnwrapper...
store = pd.HDFStore("planning_model_data.h5")
import sys
 
# setting path
sys.path.append('../')
from citylearn_wrapper import CityLearnEnvWrapper
import csv

import numpy as np
import h5py

# %%
schemas = ["../data/Test_cold_Texas/schema.json", "../data/Test_dry_Cali/schema.json",
            "../data/Test_hot_new_york/schema.json", "../data/Test_snowy_Cali_winter/schema.json"]
env_configs = [{
    "schema": schema,
    "is_evaluation": True,
} for schema in schemas]
num_episodes = 5

# %%
envs = [CityLearnEnvWrapper(env_config) for env_config in env_configs]

# %%
def collect_dataset(envs, num_episodes, save_interval = 50):
    obss = []
    next_obss = []
    actions = []
    rews = []
    dones = []
    for i in range(num_episodes):
        print(f"Episode {i}, {len(obss)} steps collected")
        env_choice = np.random.choice(len(envs))
        env = envs[env_choice]
        obs = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            next_obs, rew, done, info = env.step(action)
            #data.append([obs, next_obs, action.tolist(), rew, done])
            obss.append(obs)
            next_obss.append(next_obs)
            actions.append(action)
            rews.append([rew])
            dones.append([done])
            obs = next_obs
        if i % save_interval == save_interval - 1:
            store_data([obss, next_obss, actions, rews, dones])
            obss = []
            next_obss = []
            actions = []
            rews = []
            dones = []
    return obss, next_obss, actions, rews, dones

def convert_to_df(data, prefix):
    return pd.DataFrame(data, columns = [f"{prefix}_{i}" for i in range(len(data[0]))])

def store_data(data):
    obss, next_obss, actions, rews, dones = data
    obss_df = convert_to_df(obss, "obs")
    next_obss_df = convert_to_df(next_obss, "next_obs")
    actions_df = convert_to_df(actions, "action")
    rews_df = convert_to_df(rews, "rew")
    dones_df = convert_to_df(dones, "done")
    #df = pd.concat([obss_df, next_obss_df, actions_df, rews_df, dones_df])
    store.append("obs", obss_df, format='t')
    store.append("next_obs", next_obss_df, format='t')
    store.append("actions", actions_df, format='t')
    store.append("rews", rews_df, format='t')
    store.append("dones", dones_df, format='t')

    #pd.to_pickle(df, filename)
    # with open(filename, "w") as f:
    #     writer = csv.writer(f, delimiter='|')
    #     writer.writerow(["obs", "next_obs", "action", "rew", "done"])
    #     writer.writerows(data)

# %%
data = collect_dataset(envs, num_episodes, min(num_episodes, 50))
#store_data(data, "planning_model_data.pkl")

# %%


# %%
store.close()

# %%



