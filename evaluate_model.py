# %%
from sinergym_wrapper import SinergymWrapper
from epw_scraper.epw_data import EPW_Data
from utils import *
import numpy as np
import pandas as pd

weather_var_names = ['drybulb', 'relhum',
                        "winddir", "dirnorrad", "difhorrad"]
weather_var_rev_names = ["windspd"]

epw_data = EPW_Data.load("epw_scraper/US_epw_OU_data.pkl")
# We only need to include the default evaluation variability since we'll sample the rest later
weather_var_config = get_variability_configs(
    weather_var_names, weather_var_rev_names, epw_data=epw_data)

weather_variabilities = []
for row, pca in zip(epw_data.epw_df.iterrows(), epw_data.transformed_df.iterrows()):
    row = row[1]
    pca = pca[1]
    weather_params = {}
    for variable in weather_var_names + weather_var_rev_names:
        OU_param = np.zeros(3)
        for j in range(3):
            OU_param[j] = np.array(row[f"{variable}_{j}"]).squeeze().item()
        weather_params[variable] = OU_param
    weather_variabilities.append(weather_params)

env_config = {
    # sigma, mean, tau for OU Process
    "weather_variability": weather_variabilities,
    "variability_low": weather_var_config["train_var_low"],
    "variability_high": weather_var_config["train_var_high"],
    "use_rbc": False,
    "use_random": False,
    "sample_environments": False,
    "sinergym_timesteps_per_hour": 1
}
env = SinergymWrapper(env_config)

# %%
from uncertain_ppo import UncertainPPOTorchPolicy
# agent = UncertainPPO(config={"env_config": env_config, "env": env, "disable_env_checking": True})
checkpoints = {"activerl": "checkpoints/activerl", "vanilla": "checkpoints/vanilla"}
rews = {}
bad_idxs = {}
for name, checkpoint in checkpoints.items():
    agent = UncertainPPOTorchPolicy.from_checkpoint("checkpoints/activerl")["default_policy"]

    agent.model = agent.model.to("cuda")


    rew_df = {"rew": [], "x": [], "y": []}
    bad_idx = []
    for i in range(len(weather_variabilities)):
        try:
            pca = epw_data.transformed_df.iloc[i]
            obs = env.reset(i)
            done = False
            avg_rew = 0
            cnt = 0
            while not done:
                action = agent.compute_single_action(obs)[0]
                obs, rew, done, _ = env.step(action)
                avg_rew += rew
                cnt += 1
            rew_df["rew"].append(avg_rew / cnt)
            rew_df["x"].append(pca[0])
            rew_df["y"].append(pca[1])
        except Exception as e:
            rew_df["rew"].append(None)
            rew_df["x"].append(None)
            rew_df["y"].append(None)
            bad_idx.append(i)
    rews[name] = rew_df
    bad_idxs[name] = bad_idx
        

# %%
rew_df = pd.DataFrame.from_dict(rew_df)

# %%
import matplotlib.pyplot as plt
green = np.array([0,1,0])
red = np.array([1,0,0])
rews = (np.array(rews["activerl"]["rew"]) > np.array(rews["vanilla"]["rew"]))[:, None]

plt.scatter(rew_df["x"], rew_df["y"], c = rews * green + (1-rews) * red)
plt.savefig("test.png")



