import os
from ray.tune.logger import UnifiedLogger
from copy import deepcopy
from collections import defaultdict

MAZE_SYMBOLS = defaultdict(lambda: ".")
MAZE_SYMBOLS.update({
    b"W": "*",
    b"E": ".",
    b"G": "G",
    b"S": "P",
    b"B": "G",
})

GOAL_SYMBOLS = [b"G", b"B"]

def custom_logger_creator(log_path):

    def logger_creator(config):
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        return UnifiedLogger(config, log_path, loggers=None)

    return logger_creator

def read_gridworld(filename):
    with open(filename, 'r') as f:
        grid_str = f.read()
        grid, rew, wind_p = grid_str.split("---")
        grid_desc = eval(grid)
        rew_map = eval(rew)
        wind_p = eval(wind_p)
    return grid_desc, rew_map, wind_p

def grid_desc_to_dm(grid_desc, rew_map, wind_p):
    """Translates a simplegrid descriptor to a DM Maze descriptor.
        TODO: How can we represent wind?
    """
    subtarget_rews = []
    new_desc = []
    for i in range(len(grid_desc)):
        new_desc.append([])
        for j in range(len(grid_desc[i])):
            curr_char = bytes(grid_desc[i][j], 'utf-8')
            if curr_char in GOAL_SYMBOLS:
                subtarget_rews.append(rew_map[curr_char])
            new_desc[i].append(MAZE_SYMBOLS[curr_char])
        new_desc[i] = "".join(new_desc[i])
    new_desc = "\n".join(new_desc) + "\n"
    return new_desc, subtarget_rews

