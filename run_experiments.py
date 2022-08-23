from enum import Enum
from pathlib import Path
import gym 
from gym_simplegrid.envs import simple_grid
import matplotlib.pyplot as plt
import argparse
from citylearn.citylearn import CityLearnEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

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

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    if args.env == "cl":
        p = initialize_citylearn_params()
        env = CityLearnEnv(schema=Path("sample_schema.json"), **p)
        check_env(env) # stable baselines gym interface check

    else: 
        if args.gw_filename.strip().isnumeric():
            env = gym.make('SimpleGrid-v0', desc=int(args.gw_filename))
        else:
            grid_desc, rew_map, wind_p = read_gridworld(args.gw_filename)
            env = gym.make('SimpleGrid-v0', desc=grid_desc, reward_map = rew_map, wind_p = wind_p)

    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=30000)

    obs = env.reset()
    for i in range(16):
        action, _state = model.predict(obs, deterministic=True)
        pic = env.render(mode="ansi")
        print(pic)
        #action = int(input("- 0: LEFT - 1: DOWN - 2: RIGHT - 3: UP"))
        obs, r, done, info = env.step(action)
        if done:
            obs = env.reset()
    # plt.imsave("test.png", pic)
    