from enum import Enum
from pathlib import Path
import gym 
import torch
import torch.nn as nn
from gym_simplegrid.envs.simple_grid import SimpleGridEnvRLLib 
import matplotlib.pyplot as plt
import argparse
import ray
from citylearn.citylearn import CityLearnEnv
from uncertain_ppo import UncertainPPOTorchPolicy
from uncertain_ppo_trainer import UncertainPPO
# from state_generation import generate_states
from ray.rllib.algorithms.ppo import DEFAULT_CONFIG
from ray.rllib.models.catalog import MODEL_DEFAULTS
# from stable_baselines3 import PPO
# from stable_baselines3.common.env_checker import check_env
# from callbacks import ActiveRLCallback



class Environments(Enum):
    GRIDWORLD = "gw"
    CITYLEARN = "cl"


"""SimpleGrid is a super simple gridworld environment for OpenAI gym. It is easy to use and 
customise and it is intended to offer an environment for quick testing and prototyping 
different RL algorithms.

It is also efficient, lightweight and has few dependencies (gym, numpy, matplotlib).

SimpleGrid involves navigating a grid from Start(S) (red tile) to Goal(G) (green tile) 
without colliding with any Wall(W) (black tiles) by walking over the Empty(E) (white tiles) 
cells. The yellow circle denotes the agent's current position.

Optionally, it is possible to introduce a noise in the environment that makes the agent move 
in a random direction that can be different than the desired one.
"""


def read_gridworld(filename):
    with open(filename, 'r') as f:
        grid_str = f.read()
        grid, rew, wind_p = grid_str.split("---")
        grid_desc = eval(grid)
        rew_map = eval(rew)
        wind_p = eval(wind_p)
    return grid_desc, rew_map, wind_p

def initialize_citylearn_params():
    # Load environment
    climate_zone = 5
    buildings = ["Building_1"]

    params = {'root_directory': Path("data/Climate_Zone_" + str(climate_zone)),
            'building_attributes': 'building_attributes.json', 
            'weather_file': 'weather.csv', 
            'solar_profile': 'solar_generation_1kW.csv', 
            'carbon_intensity': 'carbon_intensity.csv',
            'building_ids': buildings,
            'buildings_states_actions': 'buildings_state_action_space.json', 
            'simulation_period': (0, 8760*1-1),
            'cost_function': ['ramping','1-load_factor','average_daily_peak','peak_demand','net_electricity_consumption','carbon_emissions'], 
            'central_agent': False,
            'save_memory': False }
    return params

def get_agent(env, env_config):
    #TODO: get dropout working
    # activation_fn = lambda : nn.Sequential(nn.Tanh(), nn.Dropout())
    # policy_kwargs = {
    #     "activation_fn": activation_fn
    # }

    config = DEFAULT_CONFIG.copy()
    config["framework"] = "torch"
    config["env"] = env
    config["env_config"] = env_config
    config["horizon"] = 100
    config["evaluation_config"]["render_env"] = True
    config["create_env_on_driver"] = True
    
    # config["model"] = MODEL_DEFAULTS
    config["env_config"]["num_dropouts_evals"] = 10
    agent = UncertainPPO(config=config)

    # agent = UncertainPPO('MlpPolicy', env, verbose=1, policy_kwargs=policy_kwargs, device="cpu")
    return agent

def train_agent(agent, timesteps, env):
    training_steps = 0
    while training_steps < timesteps:
        result = agent.train()
        training_steps = result["timesteps_total"]

def add_args(parser):
    parser.add_argument(
        "--gw_filename",
        type=str,
        help="filename to read gridworld specs from. pass an int if you want to auto generate one.",
        default="gridworlds/sample_grid.txt"
        )
    parser.add_argument(
        "--env",
        type=str,
        help="Which environment: CityLearn, GridWorld, or... future env?",
        default=Environments.GRIDWORLD.value,
        choices=[env.value for env in Environments]
    )
    parser.add_argument(
        "--num_descent_steps",
        type=int,
        help="How many steps to do gradient descent on state for Active RL",
        default=10
    )
    parser.add_argument(
        "--use_activerl",
        type=int,
        help="set to 1 to use the Active RL callback and 0 to not",
        default=0
    )

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()
    ray.init()

    if args.env == "cl":
        env = CityLearnEnv
        env_config = {
            "schema": Path("sample_schema.json")
        }

    else: 
        env = SimpleGridEnvRLLib
        env_config = {}
        if args.gw_filename.strip().isnumeric():
            env_config["desc"] = int(args.gw_filename)
        else:
            grid_desc, rew_map, wind_p = read_gridworld(args.gw_filename)
            env_config["desc"] = grid_desc
            env_config["reward_map"] = rew_map
            env_config["wind_p"] = wind_p

    agent = get_agent(env, env_config)
    # TODO: add callbacks
    # callbacks = []
    # if args.use_activerl:
    #     callbacks.append(ActiveRLCallback(num_descent_steps=args.num_descent_steps, batch_size=1, projection_fn=env.project))
    # print(agent.policy)
    
    train_agent(agent, timesteps=1, env=env)

    agent.evaluate(duration_fn=lambda x: 4 - x)


    # env = SimpleGridEnvRLLib(env_config)
    # obs = env.reset()
    # for i in range(16):
    #     action, _state = agent.compute_actions(obs)
    #     pic = env.render(mode="ansi")
    #     print(pic)
    #     #action = int(input("- 0: LEFT - 1: DOWN - 2: RIGHT - 3: UP"))
    #     obs, r, done, info = env.step(action)
    #     if done:
    #         obs = env.reset()
    # plt.imsave("test.png", pic)
    