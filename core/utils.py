from datetime import datetime
import os
from typing import Optional
from ray.tune.logger import UnifiedLogger
from copy import deepcopy
from collections import defaultdict
import cProfile
import io
import pstats

from sinergym_wrappers.epw_scraper.epw_data import EPW_Data

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
        rew_map = {str(k, "utf-8"): v for k, v in rew_map.items()}
        wind_p = eval(wind_p)
    return grid_desc, rew_map, wind_p

def grid_desc_to_dm(grid_desc, rew_map, wind_p):
    """Translates a simplegrid descriptor to a DM Maze descriptor.
        TODO: How can we represent wind?
    """
    rew_map = {bytes(k, "utf-8"): v for k, v in rew_map.items()}
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

def print_profile(profile: cProfile.Profile, log_file: Optional[str] = None, n: int = 20):
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

def get_variability_configs(names, rev_names=[], only_default_eval = False, epw_data: Optional[EPW_Data] = None):
    """
        Utility function to easily construct config arguments for the weather_variability
        related parameters of sinergym.

        :param names: Names of weather variables to add variability to
        :param rev_names: Names of weather variables to add variability that is inversely correlated with
                            the variability added to variables specified in the names param
        :param only_default_eval: Whether to return eval_variability with only the default variability or with other
                            hardcoded variabilities included.

        :return : Outputs a dictionary of variability configurations, composed of the keys "train_var", "train_var_low", "train_var_high", "eval_var"
    """
    train_variability = [build_variability_dict(names, rev_names, (1., 0., 0.001))]
    all_names =  names + rev_names
    if epw_data:
        epw_means = {name: epw_data.read_OU_param(epw_data.OU_mean, name) for name in all_names}
        epw_stds = {name: epw_data.read_OU_param(epw_data.OU_std, name) for name in all_names}
        train_variability_low = {name: tuple(epw_means[name] - 2 * epw_stds[name]) for name in all_names}
        train_variability_high = {name: tuple(epw_means[name] + 2 * epw_stds[name]) for name in all_names}
        train_variability = [{name: (var[0] * epw_stds[name], var[1] * epw_stds[name], var[2]) for name, var in train_variability[0].items()}]
    else:
        train_variability_low = {name: (0.0, -25., 0.000999) for name in names + rev_names}
        train_variability_high = {name: (15.0, 25., 0.00101) for name in names + rev_names}

    if only_default_eval:
        eval_variability = [build_variability_dict(names, rev_names, (1, 0, 0.001))]
    else:
        eval_variability = [build_variability_dict(names, rev_names, (1, 0, 0.001)),
                            build_variability_dict(names, rev_names, (1, -20, 0.001)),
                            build_variability_dict(names, rev_names, (1, 20, 0.001)),
                            build_variability_dict(names, rev_names, (10, 0, 0.001))]
    
    return {"train_var": train_variability, 
            "train_var_low": train_variability_low, 
            "train_var_high": train_variability_high,
            "eval_var": eval_variability}
    
def get_log_path(log_dir):
    now = datetime.now()
    date_time = now.strftime("%m-%d-%Y,%H-%M-%S")

    path = os.path.join(".", log_dir, date_time)
    os.makedirs(path, exist_ok=True)
    return path