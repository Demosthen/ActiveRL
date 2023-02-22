from copy import deepcopy
from core.config import ExperimentConfig
from core.utils import read_gridworld
from gridworld.callbacks import SimpleGridCallback
from gridworld.simple_grid_wrapper import SimpleGridEnvWrapper


def get_config(args):
    rllib_config = {}
    model_config = {}
    eval_env_config = {}
    env_config = {"is_evaluation": False}
    if args.gw_filename.strip().isnumeric():
        env_config["desc"] = int(args.gw_filename)
    else:
        grid_desc, rew_map, wind_p = read_gridworld(args.gw_filename)
        env_config["desc"] = grid_desc
        env_config["reward_map"] = rew_map
        env_config["wind_p"] = wind_p

    eval_env_config = deepcopy(env_config)
    eval_env_config["is_evaluation"] = True

    dummy_env = SimpleGridEnvWrapper(env_config)
    # TODO: is there a better way of counting this?
    rllib_config["evaluation_duration"] = max(
        1, args.gw_steps_per_cell) * dummy_env.nrow * dummy_env.ncol
    
    return ExperimentConfig(SimpleGridEnvWrapper, SimpleGridCallback, 
                            env_config=env_config,
                            rllib_config=rllib_config, 
                            model_config=model_config, 
                            eval_env_config=eval_env_config)