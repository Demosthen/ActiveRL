from copy import deepcopy
from run_experiments import read_gridworld
from simple_grid_wrapper import SimpleGridEnvWrapper
import gym 
import gym_simplegrid


gw_filename = "gridworlds/corridor.txt"
env_config = {"is_evaluation": False}
grid_desc, rew_map, wind_p = read_gridworld(gw_filename)
env_config["desc"] = grid_desc
env_config["reward_map"] = rew_map
env_config["wind_p"] = wind_p

env = SimpleGridEnvWrapper(env_config)

# env = gym.make('SimpleGrid-8x8-v0')
observation = env.reset()
T = 10
for i in range(T):
    action = 1
    # action = env.action_space.sample()
    env.render()
    # input()
    observation, reward, done, info = env.step(action)
    if done:
        observation = env.reset()
env.close()