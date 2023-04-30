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
import matplotlib.pyplot as plt
import torch
from writeup import run_queries
from writeup.utils import *
def add_args(parser):
    parser.add_argument(
        "--run_id",
        type=str,
        help="Run to download model artifact from. Set to RANDOM to use the random controller or RBC to use the rbc controller.",
        default=None
    )
    parser.add_argument(
        "--compare_run_id",
        type=str,
        help="Run with model artifact to compare the other run_id against. If this is not provided",
        default=None
    )
    parser.add_argument(
        "--graph_name",
        type=str,
        help="Name of graph from run_queries.py to generate",
        default=None
    )
    parser.add_argument(
        "--name",
        type=str,
        help="Name for analysis run",
        default=None
    )
    parser.add_argument(
        "--use_extreme_weather",
        action="store_true",
        help="Whether or not to use handcrafted extreme weather conditions",
        default=None
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Whether or not to overwrite existing cached reward data",
        default=None
    )
    parser.add_argument(
        "--num_repeats",
        type=int,
        help="Number of times to repeat each evaluation",
        default=None
    )
    parser.add_argument(
        "--timesteps_per_hour",
        type=int,
        help="Number of timesteps for Energyplus to run per simulated hour",
        default=1
    )    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether or not to run in debug mode"
    )    
    parser.add_argument(
        "--step",
        type=int,
        help="Which step to download the model from. If not provided, will download the latest model.",
        default=117 # ~3000000 timesteps
    )
    
    return parser
def download_model(run_id, step=None):
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
    min_artifact_id = np.inf
    artifact_id_names = {}
    artifact_ids = []
    for artifact in run.logged_artifacts():
        if "sinergym_model" in artifact.name and artifact.type == "model":
            _, artifact_id = artifact.name.split(":v")
            artifact_id = int(artifact_id)
            artifact_ids.append(artifact_id)
            artifact_id_names[artifact_id] = artifact.name
            min_artifact_id = min(min_artifact_id, artifact_id)
            if artifact_id > latest_artifact_id:
                latest_artifact_name = artifact.name
                latest_artifact_id = artifact_id
    
    order = np.argsort(artifact_ids)
    artifact_names = {}

    for i, artifact_id in enumerate(artifact_ids):
        artifact_names[i] = artifact_id_names[artifact_ids[order[i]]]
    # # Subtract min_artifact_id from all keys to get a 0-indexed dict
    # artifact_names = {k-min_artifact_id: v for k, v in artifact_names.items()}
    if step is None or step not in artifact_names:
        print(f"Step {step} not provided or not found, using latest model")
        artifact_name = latest_artifact_name
        step = latest_artifact_id - min_artifact_id
    else:
        artifact_name = artifact_names[step]
    print("Downloading artifact: ", artifact_name, " at step ", step)
    artifact = wandb.use_artifact(f"{entity}/{project}/{artifact_name}", type="model")
    artifact_dir = artifact.download(root=f"checkpoints/wandb/{run_id}")
    artifact_dir = os.path.join(artifact_dir, os.listdir(artifact_dir)[0])

    return artifact_dir

parser = argparse.ArgumentParser()
add_args(parser)
args = parser.parse_args()
DEBUG_MODE = args.debug
if (args.run_id is None and args.compare_run_id is None) and args.graph_name is None:
    raise NotImplementedError("Please specify a run id using --run_id or a graph name using --graph_name")
if __name__ == "__main__":
    #wandb.init(project="active-rl", entity="social-game-rl", config=vars(args))
    wandb.init(config=vars(args), name=args.name)

    # model1_dir = download_latest_model(args.run_id)
    # model2_dir = download_latest_model(args.compare_run_id)


weather_var_names = ['drybulb', 'relhum',
                        "winddir", "dirnorrad"]
weather_var_rev_names = ["windspd"]

epw_data = EPW_Data.load("sinergym_wrappers/epw_scraper/US_epw_OU_data.pkl")
# We only need to include the default evaluation variability since we'll sample the rest later
weather_var_config = get_variability_configs(
        weather_var_names, weather_var_rev_names, only_default_eval=False, epw_data=epw_data)

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

if DEBUG_MODE:
    weather_variabilities = weather_variabilities[:2]
base_weather_file = 'USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw'
CPU_COUNT = 24
SAMPLE_SIZE = CPU_COUNT * 5
idxs = list(range(len(weather_variabilities)))
idxs = np.concatenate([idxs[:1], np.random.choice(idxs[1:], (SAMPLE_SIZE-1,), replace=False)])
weather_variabilities = [weather_variabilities[i] for i in idxs]#weather_variabilities[idxs]
eval_weather_variabilities = weather_var_config["eval_var"] if args.use_extreme_weather else weather_variabilities
print(f"EVAL WEATHER VARIABILITIES LENGTH IS: {len(eval_weather_variabilities)}")
env_config = {
    # sigma, mean, tau for OU Process
    "weather_variability": eval_weather_variabilities,
    "variability_low": weather_var_config["train_var_low"],
    "variability_high": weather_var_config["train_var_high"],
    "use_rbc": 0, # We set these later in compute_reward
    "use_random": 0, # We set these later in compute_reward
    "sample_environments": False,
    "timesteps_per_hour": args.timesteps_per_hour,
    "act_repeat": args.timesteps_per_hour,
    "random_month": False,
    "epw_data": epw_data,
    "weather_file": base_weather_file,
    "continuous": True,
    "only_vary_offset": True,
}

# env_config = {'weather_variability': [{'drybulb': np.array([5.53173187e+00, 0.00000000e+00, 2.55034944e-03]), 'relhum': np.array([1.73128872e+01, 0.00000000e+00, 2.31712760e-03]), 'winddir': np.array([7.39984654e+01, 0.00000000e+00, 4.02298013e-04]), 'dirnorrad': np.array([3.39506556e+02, 0.00000000e+00, 9.78192172e-04]), 'windspd': np.array([1.64655725e+00, 0.00000000e+00, 3.45045547e-04])}], 'variability_low': {'drybulb': np.array([4.31066896e+00, 1.43882821e-03]), 'relhum': np.array([2.07871802e+01, 1.52442626e-03]), 'winddir': np.array([9.31378444e+01, 1.77923100e-04]), 'dirnorrad': np.array([2.26216882e+02, 3.96634341e-04]), 'windspd': np.array([1.92756975e+00, 2.60994514e-04])}, 'variability_high': {'drybulb': np.array([9.87995071e+00, 8.40623734e-03]), 'relhum': np.array([3.26129158e+01, 5.10374079e-03]), 'winddir': np.array([1.46046002e+02, 5.68863159e-04]), 'dirnorrad': np.array([3.51914077e+02, 8.28838542e-04]), 'windspd': np.array([3.73801488e+00, 8.64436358e-04])}, 'use_rbc': 0, 'use_random': 0, 'sample_environments': False, 'timesteps_per_hour': 1, 'weather_file': 'USA_AZ_Davis-Monthan.AFB.722745_TMY3.epw', 'epw_data': epw_data, 'continuous': True, 'random_month': False}
# %%
from core.uncertain_ppo.uncertain_ppo import UncertainPPOTorchPolicy
def compute_reward(checkpoint, i, seed, tag):
    np.random.seed(seed)
    torch.manual_seed(seed)
    env_config["use_rbc"] = (checkpoint == "RBC")
    env_config["use_random"] = (checkpoint == "RANDOM")
    env = SinergymWrapper(env_config)
    if checkpoint in ["RBC", "RANDOM"]:
        compute_action = lambda obs: 0 # it doesn't matter what we send in because it will get replaced with the appropriate action anyway
    else:
        loaded_agent = False
        fail_cnt = 0
        while not loaded_agent:
            try:
                agent = UncertainPPOTorchPolicy.from_checkpoint(checkpoint)["default_policy"]
                loaded_agent = True
            except:
                fail_cnt += 1
                print(f"Failed to load agent {fail_cnt}, trying again...")
        #agent = UncertainPPOTorchPolicy.from_checkpoint(checkpoint)["default_policy"]
        compute_action = lambda obs: np.array(agent.compute_single_action(obs)[0])

    # agent.model = agent.model.to("cuda")
    rew_df = {"rew": [], "x": [], "y": [], "idx": []}
    bad_idx = []
    try:
        obs = env.reset(i)
        done = False
        avg_rew = 0
        cnt = 0
        while not done and (cnt < 10 or not DEBUG_MODE):
            action = compute_action(obs)
            action = np.clip(action, -1, 1) # to be consistent with RLLib's automatic action normalization
            obs, rew, done, info = env.step(action)
            avg_rew += rew
            cnt += 1
        rew_df["rew"].append(avg_rew / cnt)

        pca = epw_data.transformed_df.iloc[i]
        rew_df["x"].append(pca[0])
        rew_df["y"].append(pca[1])
        print(cnt)
    except Exception as e:
        print(e)
        print(tag)
        rew_df["rew"].append(None)
        rew_df["x"].append(None)
        rew_df["y"].append(None)
        bad_idx.append(i)
    rew_df["idx"].append(i)
    env.close()
    return rew_df, bad_idx




def get_checkpoints(graph_name):
    print("Downloading checkpoints")
    api = wandb.Api(timeout=20)
    # Project is specified by <entity/project-name>
    runs = api.runs("social-game-rl/active-rl",
        run_queries.QUERIES[graph_name]
        )
    api.runs("social-game-rl/active-rl")
    GROUP_BY = run_queries.GROUP_BY[graph_name]
    checkpoints = {}
    for run in runs:       

        for tag in run.tags:
            if tag not in GROUP_BY:
                continue
            
            run_id = run.id
            print(f"Downloading checkpoint for run {run_id} with tag {tag}")
            model_dir = download_model(run_id, args.step)
            if tag not in checkpoints:
                checkpoints[tag] = [(run_id, model_dir)]
            else:
                checkpoints[tag].append((run_id, model_dir))
    return checkpoints


if __name__ == "__main__":
    
    import time
    start = time.perf_counter()
    # agent = UncertainPPO(config={"env_config": env_config, "env": env, "disable_env_checking": True})
    #checkpoints = {args.run_id: model1_dir, args.compare_run_id: model2_dir}#{"activerl": "checkpoints/activerl", "vanilla": "checkpoints/vanilla"}
    checkpoints = get_checkpoints(args.graph_name)
    agents = {}
    rews = {}
    bad_idxs = {}
    names = {}

    # compute_reward(model1_dir, 0)
    ctx = mp.get_context('spawn')
    save_dir = f"checkpoints/comparisons/"
    prefix = "extreme_" if args.use_extreme_weather else ""
    prefix += f"{args.timesteps_per_hour}_"
    prefix += f"{args.step}_" if args.step else ""
    idxs = list(range(0, len(eval_weather_variabilities), 1))
    

    with ctx.Pool(CPU_COUNT) as workers:
        for tag, checkpoint_list in checkpoints.items():
            for name, checkpoint in checkpoint_list:
                for k in range(args.num_repeats):
                    ckpt_save_file = os.path.join(save_dir, f"{prefix}_{name}_{k}.pkl")
                    loaded_file = False
                    if os.path.exists(ckpt_save_file) and not args.no_cache:
                        with open(ckpt_save_file, 'rb') as f:
                            rew_df, bad_idx = pickle.load(f)
                        loaded_file = True
                    else:
                        
                        rew_args = [(checkpoint, idx, k, tag) for idx in idxs]
                        rew_df = {"rew": [], "x": [], "y": [], "idx": []}
                        bad_idx = []
                        for result in workers.starmap(compute_reward, rew_args):
                            rew_df["rew"].extend(result[0]["rew"])
                            rew_df["x"].extend(result[0]["x"])
                            rew_df["y"].extend(result[0]["y"])
                            rew_df["idx"].extend(result[0]["idx"])
                            bad_idx.extend(result[1])
                        rew_df = pd.DataFrame.from_dict(rew_df)
                        
                        if not DEBUG_MODE:
                            with open(ckpt_save_file, 'wb') as f:
                                pickle.dump((rew_df, bad_idx), f)
                    if tag not in rews:
                        rews[tag] = [rew_df]
                        bad_idxs[tag] = bad_idx
                    else:
                        rews[tag].append(rew_df)
                        bad_idxs[tag].extend(bad_idx)
        
    for tag in checkpoints.keys():
        rews[tag] = pd.concat(rews[tag])
    save_file = os.path.join(save_dir, f"{prefix}{args.graph_name}.pkl")
    os.makedirs(save_dir, exist_ok=True)
    with open(save_file, "wb") as f:
        pickle.dump((rews, bad_idxs), f)

    rew_artifact = wandb.Artifact("reward_data", type="evaluation")
    rew_artifact.add_file(save_file)
    wandb.log_artifact(rew_artifact)

    # %%
    if args.use_extreme_weather:
        plot_bars(start, rews, bad_idxs, graph_name=args.graph_name)
    else:
        plot_scatter(start, rews, bad_idxs)


