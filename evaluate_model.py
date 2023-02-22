# %%
import pickle
from sinergym_wrapper import SinergymWrapper
from epw_scraper.epw_data import EPW_Data
from utils import *
import numpy as np
import pandas as pd
from multiprocessing import Pool

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

# %%
from uncertain_ppo import UncertainPPOTorchPolicy
def compute_reward(checkpoint, i):
    env = SinergymWrapper(env_config)
    agent = UncertainPPOTorchPolicy.from_checkpoint(checkpoint)["default_policy"]

    agent.model = agent.model.to("cuda")
    rew_df = {"rew": [], "x": [], "y": [], "idx": []}
    bad_idx = []
    # try:
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
    # except Exception as e:
    #     rew_df["rew"].append(None)
    #     rew_df["x"].append(None)
    #     rew_df["y"].append(None)
    #     bad_idx.append(i)
    rew_df["idx"].append(i)
    return rew_df, bad_idx

if __name__ == "__main__":
    import time
    start = time.perf_counter()
    # agent = UncertainPPO(config={"env_config": env_config, "env": env, "disable_env_checking": True})
    checkpoints = {"activerl": "checkpoints/activerl", "vanilla": "checkpoints/vanilla"}
    agents = {}
    rews = {}
    bad_idxs = {}


    with Pool(6) as workers:
        for name, checkpoint in checkpoints.items():
            idxs = range(0, len(weather_variabilities))
            args = [(checkpoint, idx) for idx in idxs]
            rew_df = {"rew": [], "x": [], "y": [], "idx": []}
            bad_idx = []
            for result in workers.starmap(compute_reward, args):
                rew_df["rew"].extend(result[0]["rew"])
                rew_df["x"].extend(result[0]["x"])
                rew_df["y"].extend(result[0]["y"])
                rew_df["idx"].extend(result[0]["idx"])
                bad_idx.extend(result[1])
            rew_df = pd.DataFrame.from_dict(rew_df)
            rews[name] = rew_df
            bad_idxs[name] = bad_idx

    with open("checkpoints/reward_data.pkl", "wb") as f:
        pickle.dump((rews, bad_idxs), f)

    # %%
    import matplotlib.pyplot as plt
    green = np.array([0,1,0])
    red = np.array([1,0,0])
    all_bad_idxs = set(bad_idxs["activerl"] + bad_idxs["vanilla"])
    print(all_bad_idxs)
    # drop all bad indexes and sort by index
    activerl_dfs = rews["activerl"]
    print(activerl_dfs)
    idxs = ~activerl_dfs["idx"].isin(all_bad_idxs)
    activerl_dfs = activerl_dfs[idxs].sort_values("idx")
    vanilla_dfs = rews["vanilla"]
    vanilla_dfs = vanilla_dfs[idxs].sort_values("idx")

    rews = (np.array(activerl_dfs["rew"]) > np.array(vanilla_dfs["rew"]))[:, None]
    xs = vanilla_dfs["x"]
    ys = vanilla_dfs["y"]
    end = time.perf_counter()
    print(f"TOOK {end-start} SECONDS")

    plt.scatter(xs, ys, c = rews * green + (1-rews) * red)
    plt.savefig("test.png")
    breakpoint()



