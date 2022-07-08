import gym 
import gym_simplegrid
import argparse

"""SimpleGrid is a super simple gridworld environment for OpenAI gym. It is easy to use and customise and it is intended to offer an environment for quick testing and prototyping different RL algorithms.

It is also efficient, lightweight and has few dependencies (gym, numpy, matplotlib).



SimpleGrid involves navigating a grid from Start(S) (red tile) to Goal(G) (green tile) without colliding with any Wall(W) (black tiles) by walking over the Empty(E) (white tiles) cells. The yellow circle denotes the agent's current position.

Optionally, it is possible to introduce a noise in the environment that makes the agent move in a random direction that can be different than the desired one."""
def read_gridworld(filename):
    with open(filename, 'r') as f:
        grid_str = f.read()
        grid, rew = grid_str.split("---")
        grid_desc = eval(grid)
        rew_map = eval(rew)
    return grid_desc, rew_map

def add_args(parser):
    parser.add_argument(
        "--filename",
        type=str,
        help="filename to read gridworld specs from",
        default="gridworlds/sample_grid.txt"
        )
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    grid_desc, rew_map = read_gridworld(args.filename)
    env = gym.make('SimpleGrid-v0', desc=grid_desc, reward_map = rew_map)
    env.reset()
    pic = env.render(mode="ansi")
    print(pic)
    