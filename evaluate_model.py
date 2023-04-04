# %%
import sys

import PIL
sys.path.append('../')
import pickle
from sinergym_wrappers.sinergym_wrapper import SinergymWrapper
from sinergym_wrappers.epw_scraper.epw_data import EPW_Data
from core.utils import *
import numpy as np
import pandas as pd
from itertools import starmap
import multiprocessing as mp
import wandb
import argparse
def add_args(parser):
    parser.add_argument(
        "--run_id",
        type=str,
        help="Run to download model artifact from",
        default=None
    )
    parser.add_argument(
        "--compare_run_id",
        type=str,
        help="Run with model artifact to compare the other run_id against. If this is not provided",
        default=None
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Name for analysis run",
        default=None
    )
    
    return parser
def download_latest_model(run_id):
    """
    Download the latest 'sinergym_model' artifact from a specified Weights & Biases run.

    This function searches for the latest version of the 'sinergym_model' artifact of type 'model'
    in a specific W&B run, downloads it, and returns the local directory path where the artifact
    has been downloaded.

    Args:
        run_id (str): A string that should be
            the run_id of the target Weights & Biases run.

    Returns:
        str: The local directory path where the latest 'sinergym_model' artifact has been downloaded.
    """
    entity = "social-game-rl"
    project = "active-rl"
    run = wandb.Api().run(f"{entity}/{project}/{run_id}")
    latest_artifact_id = -1
    latest_artifact_name = ""
    for artifact in run.logged_artifacts():
        if "sinergym_model" in artifact.name and artifact.type == "model":
            _, artifact_id = artifact.name.split(":v")
            artifact_id = int(artifact_id)
            if artifact_id > latest_artifact_id:
                latest_artifact_name = artifact.name
                latest_artifact_id = artifact_id
    artifact = wandb.use_artifact(f"{entity}/{project}/{latest_artifact_name}", type="model")
    artifact_dir = artifact.download(root=f"checkpoints/wandb/{run_id}")
    artifact_dir = os.path.join(artifact_dir, os.listdir(artifact_dir)[0])

    return artifact_dir

parser = argparse.ArgumentParser()
add_args(parser)
args = parser.parse_args()
if args.run_id is None or args.compare_run_id is None:
    raise NotImplementedError("Please specify a run id using --run_id")
if __name__ == "__main__":
    #wandb.init(project="active-rl", entity="social-game-rl", config=vars(args))
    wandb.init(config=vars(args), name=args.name)

    model1_dir = download_latest_model(args.run_id)
    model2_dir = download_latest_model(args.compare_run_id)


weather_var_names = ['drybulb', 'relhum',
                        "winddir", "dirnorrad"]
weather_var_rev_names = ["windspd"]

epw_data = EPW_Data.load("sinergym_wrappers/epw_scraper/US_epw_OU_data.pkl")
# We only need to include the default evaluation variability since we'll sample the rest later
weather_var_config = get_variability_configs(
        weather_var_names, weather_var_rev_names, only_default_eval=True, epw_data=epw_data)

weather_variabilities = []#weather_var_config["train_var"] # start with the training config
min_diff = 100000
for row, pca in zip(epw_data.epw_df.iterrows(), epw_data.transformed_df.iterrows()):
    row = row[1]
    pca = pca[1]
    weather_params = {}
    diff = 0
    for variable in weather_var_names + weather_var_rev_names:
        OU_param = np.zeros(3)
        for j in range(3):
            OU_param[j] = np.array(row[f"{variable}_{j}"]).squeeze().item()
        weather_params[variable] = OU_param
        diff += np.mean(np.abs(OU_param))
    weather_variabilities.append(weather_params)
    min_diff = min(diff, min_diff)
print(min_diff)

base_weather_file = 'USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw'
env_config = {
    # sigma, mean, tau for OU Process
    "weather_variability": weather_variabilities,
    "variability_low": weather_var_config["train_var_low"],
    "variability_high": weather_var_config["train_var_high"],
    "use_rbc": 0,
    "use_random": 0,
    "sample_environments": False,
    "sinergym_timesteps_per_hour": 1,
    "random_month": False,
    "epw_data": epw_data,
    "weather_file": base_weather_file,
    "continuous": True
}

# env_config = {'weather_variability': [{'drybulb': np.array([5.53173187e+00, 0.00000000e+00, 2.55034944e-03]), 'relhum': np.array([1.73128872e+01, 0.00000000e+00, 2.31712760e-03]), 'winddir': np.array([7.39984654e+01, 0.00000000e+00, 4.02298013e-04]), 'dirnorrad': np.array([3.39506556e+02, 0.00000000e+00, 9.78192172e-04]), 'windspd': np.array([1.64655725e+00, 0.00000000e+00, 3.45045547e-04])}], 'variability_low': {'drybulb': np.array([4.31066896e+00, 1.43882821e-03]), 'relhum': np.array([2.07871802e+01, 1.52442626e-03]), 'winddir': np.array([9.31378444e+01, 1.77923100e-04]), 'dirnorrad': np.array([2.26216882e+02, 3.96634341e-04]), 'windspd': np.array([1.92756975e+00, 2.60994514e-04])}, 'variability_high': {'drybulb': np.array([9.87995071e+00, 8.40623734e-03]), 'relhum': np.array([3.26129158e+01, 5.10374079e-03]), 'winddir': np.array([1.46046002e+02, 5.68863159e-04]), 'dirnorrad': np.array([3.51914077e+02, 8.28838542e-04]), 'windspd': np.array([3.73801488e+00, 8.64436358e-04])}, 'use_rbc': 0, 'use_random': 0, 'sample_environments': False, 'timesteps_per_hour': 1, 'weather_file': 'USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw', 'epw_data': epw_data, 'continuous': True, 'random_month': False}
# %%
from core.uncertain_ppo.uncertain_ppo import UncertainPPOTorchPolicy
def compute_reward(checkpoint, i):
    env = SinergymWrapper(env_config)
    agent = UncertainPPOTorchPolicy.from_checkpoint(checkpoint)["default_policy"]

    agent.model = agent.model.to("cuda")
    rew_df = {"rew": [], "x": [], "y": [], "idx": []}
    bad_idx = []
    try:
        pca = epw_data.transformed_df.iloc[i]
        obs = env.reset(i)
        done = False
        avg_rew = 0
        cnt = 0
        while not done:
            action = np.array(agent.compute_single_action(obs)[0])
            action = np.clip(action, -1, 1)
            obs, rew, done, _ = env.step(action)
            avg_rew += rew
            cnt += 1
        rew_df["rew"].append(avg_rew / cnt)
        rew_df["x"].append(pca[0])
        rew_df["y"].append(pca[1])
    except Exception as e:
        print(e)
        rew_df["rew"].append(None)
        rew_df["x"].append(None)
        rew_df["y"].append(None)
        bad_idx.append(i)
    rew_df["idx"].append(i)
    env.close()
    return rew_df, bad_idx

if __name__ == "__main__":
    import time
    start = time.perf_counter()
    # agent = UncertainPPO(config={"env_config": env_config, "env": env, "disable_env_checking": True})
    checkpoints = {args.run_id: model1_dir, args.compare_run_id: model2_dir}#{"activerl": "checkpoints/activerl", "vanilla": "checkpoints/vanilla"}
    agents = {}
    rews = {}
    bad_idxs = {}

    # compute_reward(model1_dir, 0)
    ctx = mp.get_context('spawn')
    with ctx.Pool(8) as workers:
        for name, checkpoint in checkpoints.items():
            idxs = range(0, len(weather_variabilities), 1)
            rew_args = [(checkpoint, idx) for idx in idxs]
            rew_df = {"rew": [], "x": [], "y": [], "idx": []}
            bad_idx = []
            for result in workers.starmap(compute_reward, rew_args):
                rew_df["rew"].extend(result[0]["rew"])
                rew_df["x"].extend(result[0]["x"])
                rew_df["y"].extend(result[0]["y"])
                rew_df["idx"].extend(result[0]["idx"])
                bad_idx.extend(result[1])
            rew_df = pd.DataFrame.from_dict(rew_df)
            rews[name] = rew_df
            bad_idxs[name] = bad_idx
    save_dir = f"checkpoints/comparisons/"
    save_file = os.path.join(save_dir, "{args.run_id}_{args.compare_run_id}.pkl")
    os.makedirs(save_dir, exist_ok=True)
    with open(save_file, "wb") as f:
        pickle.dump((rews, bad_idxs), f)

    rew_artifact = wandb.Artifact("reward_data", type="evaluation")
    rew_artifact.add_file(save_file)
    wandb.log_artifact(rew_artifact)

    # %%
    import matplotlib.pyplot as plt
    green = np.array([0,1,0])
    red = np.array([1,0,0])
    all_bad_idxs = set(bad_idxs[args.run_id] + bad_idxs[args.compare_run_id])
    print(all_bad_idxs)
    # drop all bad indexes and sort by index
    run_dfs = rews[args.run_id]
    print(f"{args.run_id}: ", run_dfs)
    idxs = ~run_dfs["idx"].isin(all_bad_idxs)
    run_dfs = run_dfs[idxs].sort_values("idx")
    compare_dfs = rews[args.compare_run_id]
    compare_dfs = compare_dfs[idxs].sort_values("idx")
    print(f"{args.compare_run_id}: ", compare_dfs)

    rews = (np.array(run_dfs["rew"]) > np.array(compare_dfs["rew"]))[:, None]
    xs = compare_dfs["x"]
    ys = compare_dfs["y"]
    end = time.perf_counter()
    print(f"TOOK {end-start} SECONDS")

    plt.scatter(xs[1:], ys[1:], c = rews[1:] * green + (1-rews)[1:] * red, s=10)
    plt.scatter(xs[:1], ys[:1], c = rews[:1] * green + (1-rews)[:1] * red, marker="*", s=60, edgecolors="black")
    plt.savefig("test.png")

    wandb.log({"viz": wandb.Image("test.png")})


