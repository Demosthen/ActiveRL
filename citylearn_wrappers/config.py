from copy import deepcopy
from pathlib import Path
from citylearn_wrappers.callbacks import CitylearnCallback
from citylearn_wrappers.citylearn_wrapper import CityLearnEnvWrapper
from core.config import ExperimentConfig
from run_experiments import CL_EVAL_PATHS

def get_config(args):
    rllib_config = {}
    model_config = {}
    eval_env_config = {}

    env_config = {
        "schema": Path(args.cl_filename),
        "planning_model_ckpt": args.planning_model_ckpt,
        "is_evaluation": False,
        "use_rbc_residual": args.cl_use_rbc_residual,
        "action_multiplier": args.cl_action_multiplier
    }

    eval_env_config = deepcopy(env_config)
    eval_env_config["schema"] = [Path(filename)
                                    for filename in CL_EVAL_PATHS]
    eval_env_config["is_evaluation"] = True

    rllib_config["horizon"] = args.horizon
    rllib_config["evaluation_duration"] = len(CL_EVAL_PATHS)
    rllib_config["soft_horizon"] = False
    
    return ExperimentConfig(CityLearnEnvWrapper, CitylearnCallback, 
                            env_config=env_config,
                            rllib_config=rllib_config, 
                            model_config=model_config, 
                            eval_env_config=eval_env_config)