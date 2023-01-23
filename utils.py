import os
from ray.tune.logger import UnifiedLogger
from copy import deepcopy
from collections import defaultdict
import cProfile
import io
import pstats

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

def print_profile(profile: cProfile.Profile, log_file: str = None, n: int = 20):
    """
    Prints profiling results to stdout, sorted by total amount of time spent in each function,
    and to log file if specified.
    :profile: the actual cProfile.Profile instance
    :log_file: path to where to log the profiling stats
    :n: how many lines to print
    """
    s = io.StringIO()
    sortby = "cumulative"  # Ordered
    ps = pstats.Stats(profile, stream=s).strip_dirs().sort_stats(sortby)
    ps.print_stats(n)
    print(s.getvalue())
    if log_file is not None:
        with open(log_file, "w") as f:
            f.write(s.getvalue())
        # profile.dump_stats(log_file)

def build_variability_dict(names, rev_names, variability):
    """ 
        Builds a weather variability dictionary that applies the specified
        variability to the weather variables names provided, with the offset parameter
        negated for variables provided in rev_names. For example, 
        build_variability_dict(names=["a", "b"], rev_names=["c"], (1, 20, 0.001))
        outputs {"a": (1, 20, 0.001), "b": (1, 20, 0.001), "c": (1, -20, 0.001)}
    """
    ret = {}

    # Need to make it a list since tuple is not mutable
    rev_variability = list(deepcopy(variability))
    rev_variability[1] *= -1
    rev_variability = tuple(rev_variability)

    for name in names:
        ret[name] = variability
    
    for name in rev_names:
        ret[name] = rev_variability
    return ret