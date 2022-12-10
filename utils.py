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
    # Initialize new_desc, reserving some space for a wall at the beginning
    new_desc = []
    for i in range(len(grid_desc)):
        # Automatically add wall at beginning of row
        new_desc.append(["*"])
        for j in range(len(grid_desc[i])):
            curr_char = bytes(grid_desc[i][j], 'utf-8')
            if curr_char in GOAL_SYMBOLS:
                subtarget_rews.append(rew_map[curr_char])
            new_desc[i].append(MAZE_SYMBOLS[curr_char])
        # Automatically add wall at end of row 
        new_desc[i].append("*")
        new_desc[i] = "".join(new_desc[i])

    # Add wall rows at top and bottom of grid
    width = len(new_desc[0])
    wall_row = "".join(["*"] * width)
    new_desc = [wall_row] + new_desc + [wall_row]

    # Put everything into a string separated by newlines
    new_desc = "\n".join(new_desc) + "\n"
    return new_desc, subtarget_rews

def states_to_np(state, inplace=True):
    if not inplace:
        state = deepcopy(state)
    if isinstance(state, dict):
        for k, v in state.items():
            state[k] = v.detach().squeeze().cpu().numpy()
        return state
    else:
        return state.detach().squeeze().cpu().numpy()

def flatten_dict_of_lists(d):
    out = []
    for k, v in d.items():
        out.extend(v)
    return out