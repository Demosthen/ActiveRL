from copy import deepcopy

import numpy as np
from core.config import ExperimentConfig
from core.constants import RESPAWNABLE_TOKENS
from core.utils import flatten_dict_of_lists, grid_desc_to_dm, read_gridworld
from dm_maze_wrappers.callbacks import DMMazeCallback
from dm_maze_wrappers.dm_wrapper import DM_Maze_Obs_Wrapper
from dm_maze_wrappers.model import ComplexInputNetwork

def get_config(args):
    rllib_config = {}
    model_config = {}
    eval_env_config = {}
    
    grid_desc, rew_map, wind_p = read_gridworld(args.dm_filename)
    maze_str, subtarget_rews = grid_desc_to_dm(grid_desc, rew_map, wind_p)
    env = DM_Maze_Obs_Wrapper
    env_config = {
        "maze_str": maze_str,
        "subtarget_rews": subtarget_rews,
        "random_state": np.random.RandomState(42),
        "strip_singleton_obs_buffer_dim": True,
        "time_limit": args.horizon * args.control_timestep,
        "aliveness_reward": args.aliveness_reward,
        "distance_reward_scale": args.distance_reward_scale,
        "use_all_geoms": args.use_all_geoms,
        "walker": args.walker,
        "control_timestep": args.control_timestep,
        "physics_timestep": args.physics_timestep
    }

    eval_env_config = deepcopy(env_config)

    dummy_env = env(env_config)

    model_config["dim"] = 32
    model_config["conv_filters"] = [
        [8, [8, 8], 4],
        [16, [3, 3], 2],
        [32, [8, 8], 1],
    ]
    model_config["post_fcnet_hiddens"] = [64, 64]
    model_config["custom_model"] = ComplexInputNetwork

    rllib_config["evaluation_duration"] = 1
    rllib_config["horizon"] = args.horizon
    rllib_config["keep_per_episode_custom_metrics"] = False
    rllib_config["batch_mode"] = "truncate_episodes"
    rllib_config["evaluation_sample_timeout_s"] = 600
    rllib_config["rollout_fragment_length"] = "auto"#1000
    dummy_arena = dummy_env.get_task()._maze_arena
    grid_positions = flatten_dict_of_lists(
        dummy_arena.find_token_grid_positions(RESPAWNABLE_TOKENS))
    rllib_config["evaluation_duration"] = args.dm_steps_per_cell
    rllib_config["evaluation_parallel_to_training"] = True
    full_eval_duration = max(
        1, args.dm_steps_per_cell) * len(grid_positions)

    def full_eval_fn(agent): return agent.evaluate(
        lambda x: full_eval_duration - x)
    # rllib_config["record_env"] = True
    return ExperimentConfig(DM_Maze_Obs_Wrapper, DMMazeCallback, 
                            env_config=env_config,
                            full_eval_fn=full_eval_fn,
                            rllib_config=rllib_config, 
                            model_config=model_config, 
                            eval_env_config=eval_env_config)