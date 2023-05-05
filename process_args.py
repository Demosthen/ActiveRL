import os

from core.constants import Environments, SG_WEATHER_TYPES

def add_args(parser):
    # GENERAL PARAMS
    parser.add_argument(
        "--num_gpus",
        type=int,
        help="number of gpus to use, default = 1",
        default=1
    )

    # LOGGING PARAMS
    parser.add_argument(
        "--log_path",
        type=str,
        help="filename to read gridworld specs from. pass an int if you want to auto generate one.",
        default="logs"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="whether or not to log with wandb"
    )
    parser.add_argument(
        "--project",
        type=str,
        help="wandb project to send metrics to",
        default="active-rl"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Whether to profile this run to debug performance"
    )
    # GENERAL ENV PARAMS
    parser.add_argument(
        "--env",
        type=str,
        help="Which environment: CityLearn, GridWorld, Deepmind Maze or... future env?",
        default=Environments.GRIDWORLD.value,
        choices=[env.value for env in Environments]
    )
    parser.add_argument(
        "--num_timesteps",
        type=int,
        help="Number of timesteps to collect from environment during training",
        default=5000
    )

    # ALGORITHM PARAMS
    parser.add_argument(
        "--train_batch_size",
        type=int,
        help="Size of training batch",
        default=256
    )
    parser.add_argument(
        "--horizon", "--soft_horizon",
        type=int,
        help="Horizon of timesteps to compute reward over (WARNING, I don't think this does anything for dm_control maze)",
        default=48
    )
    parser.add_argument(
        "--clip_param",
        type=float,
        help="PPO clipping parameter",
        default=0.3
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate for PPO",
        default=5e-5
    )
    parser.add_argument(
        "--gamma",
        type=float,
        help="Gamma for PPO",
        default=0.99
    )
    parser.add_argument(
        "--num_sgd_iter",
        type=int,
        help="num_sgd_iter for PPO",
        default=30
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        help="Number of training steps between evaluation runs",
        default=1
    )
    parser.add_argument(
        "--num_training_workers",
        type=int,
        help="Number of workers to collect data during training",
        default=3
    )
    parser.add_argument(
        "--num_eval_workers",
        type=int,
        help="Number of workers to collect data during eval",
        default=1
    )
    parser.add_argument(
        "--num_envs_per_worker",
        type=int,
        help="Number of workers to collect data during training",
        default=1
    )
    # CITYLEARN ENV PARAMS
    parser.add_argument(
        "--cl_filename",
        type=str,
        help="schema file to read citylearn specs from.",
        default="./data/citylearn_challenge_2022_phase_1/schema.json"
    )
    parser.add_argument(
        "--cl_eval_folder",
        type=str,
        default="./data/all_buildings",
        help="Which folder\'s building files to evaluate with",
    )
    parser.add_argument(
        "--cl_use_rbc_residual",
        type=int,
        default=0,
        help="Whether or not to train actions as residuals on the rbc"
    )
    parser.add_argument(
        "--cl_action_multiplier",
        type=float,
        default=1,
        help="A scalar to scale the agent's outputs by"
    )

    # GRIDWORLD ENV PARAMS
    parser.add_argument(
        "--gw_filename",
        type=str,
        help="filename to read gridworld specs from. pass an int if you want to auto generate one.",
        default="gridworlds/sample_grid.txt"
    )
    parser.add_argument(
        "--gw_steps_per_cell",
        type=int,
        help="number of times to evaluate each cell, min=1",
        default=1
    )

    # DM MAZE ENV PARAMS
    parser.add_argument(
        "--dm_filename",
        type=str,
        help="filename to read gridworld specs from. pass an int if you want to auto generate one.",
        default="gridworlds/sample_grid.txt"
    )

    parser.add_argument(
        "--aliveness_reward",
        type=float,
        help="Reward to give agent for staying alive",
        default=0.01
    )

    parser.add_argument(
        "--distance_reward_scale",
        type=float,
        help="Scale of reward for agent getting closer to goal",
        default=0.01
    )

    parser.add_argument(
        "--use_all_geoms",
        action="store_true",
        help="Whether to use all possible geoms or try to consolidate them for faster speed (if you turn this on things will slow down significantly)",
    )

    parser.add_argument(
        "--walker",
        type=str,
        choices=["ant", "ball"],
        help="What type of walker to use. Ant and ball are currently supported",
        default="ant"
    )
    parser.add_argument(
        "--dm_steps_per_cell",
        type=int,
        help="number of times to evaluate each cell, min=1",
        default=1
    )
    parser.add_argument(
        "--control_timestep",
        type=float,
        help="Time between control timesteps in seconds",
        default=0.1  # DEFAULT_CONTROL_TIMESTEP
    )
    parser.add_argument(
        "--physics_timestep",
        type=float,
        help="Time between physics timesteps in seconds",
        default=0.02  # DEFAULT_PHYSICS_TIMESTEP
    )

    # SINERGYM ENV PARAMS
    parser.add_argument(
        "--use_rbc",
        type=int,
        help="Whether or not to override all actions with that of Rule Based Controller (RBC). Set to 1 to enable.",
        default=0
    )
    parser.add_argument(
        "--use_random",
        type=int,
        help="Whether or not to override all actions with that of Random Controller. Set to 1 to enable.",
        default=0
    )
    parser.add_argument(
        "--only_drybulb",
        action="store_true",
        help="Whether to restrict the weather variability changes to only drybulb outdoor temperature",
    )
    parser.add_argument(
        "--sample_envs",
        action="store_true",
        help="Whether to randomly sample environments from the US",
    )
    parser.add_argument(
        "--sinergym_timesteps_per_hour",
        type=int,
        default=1,
        help="How many timesteps to have in each hour",
    )
    parser.add_argument(
        "--eval_fidelity_ratio",
        type=int,
        default=1,
        help="Ratio of timesteps for Energy Plus to simulate building during evaluation compared to training for each hour",
    )
    parser.add_argument(   
        "--base_weather",
        type=str,
        default=SG_WEATHER_TYPES.HOT.value,
        help = "What kind of base weather to have",
        choices=[weather.value for weather in SG_WEATHER_TYPES]
    )
    parser.add_argument(   
        "--random_month",
        action="store_true",
        help = "Whether to sample a random week's weather instead of a whole year",
    )
    parser.add_argument(   
        "--no_noise",
        action="store_true",
        help = "Whether to have any noise in the training weather or not",
    )
    parser.add_argument(   
        "--continuous",
        action="store_true",
        help = "Whether to use a continuous action space or not",
    )
    parser.add_argument(   
        "--only_vary_offset",
        action="store_true",
        help = "Whether to only vary the offset of the weather or also the scale and time constant",
    )

    # ACTIVE RL PARAMS
    parser.add_argument(
        "--use_activerl",
        type=float,
        help="Probability of a training episode using an active start. Set to 1 to only use the active start and 0 to use default",
        default=0
    )
    parser.add_argument(
        "--use_random_reset",
        type=float,
        help="Whether or not to reset the environment to random states",
        default=0
    )
    parser.add_argument(
        "--num_descent_steps",
        type=int,
        help="How many steps to do gradient descent on state for Active RL",
        default=10
    )
    parser.add_argument(
        "--no_coop",
        action="store_true",
        help="Whether or not to use the constrained optimizer for Active RL optimization"
    )
    parser.add_argument(
        "--planning_model_ckpt",
        type=str,
        help="File path to planning model checkpoint. Leave as None to not use the planning model",
        default=None
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to pass in to rllib workers",
        default=12345678
    )
    parser.add_argument(
        "--planning_uncertainty_weight",
        type=float,
        help="What relative weight to give to the planning uncertainty compared to agent uncertainty",
        default=1
    )
    parser.add_argument(
        "--activerl_lr",
        type=float,
        help="Learning rate for Active RL",
        default=0.1
    )
    parser.add_argument(
        "--activerl_reg_coeff",
        type=float,
        help="Weighting coefficient penalizing generated state's distance from original state",
        default=0.01
    )
    parser.add_argument(
        "--num_dropout_evals",
        type=int,
        help="Number of dropout evaluations to run to estimate uncertainty",
        default=5
    )
    parser.add_argument(
        "--plr_d",
        type=float,
        help="Set to 1 to turn on PLR and 0 to turn it off",
        default=0.0
    )
    parser.add_argument(
        "--plr_beta",
        type=float,
        help="Beta parameter for PLR",
        default=0.1
    )
    parser.add_argument(
        "--plr_envs_to_1",
        type=int,
        help="Number of environments to go through until probability of using PLR is 1",
        default=100
    )
    parser.add_argument(
        "--env_repeat",
        type=int,
        help="Number of train steps to repeat the same env parameters",
        default=1
    )
    parser.add_argument(
        "--start",
        type=int,
        help="Number of train steps after which to start using any smart reset methods",
        default=0
    )
    parser.add_argument(
        "--plr_rho",
        type=float,
        help="Staleness mixing parameter for PLR",
        default=0.1
    )
    parser.add_argument(
        "--dropout",
        type=float,
        help="Dropout parameter",
        default=0.5
    )
    parser.add_argument(
        "--full_eval_interval",
        type=int,
        help="Number of training steps between full evaluation runs (i.e. visiting every state in gridworld or mujoco maze)",
        default=10
    )
    # Combo args for wandb sweeps
    parser.add_argument(
        "--sinergym_sweep",
        type=str,
        help="Sets use_activerl, use_rbc, use_random, and use_random_reset, respectively, all at the same time. Pass in the values as a comma delimited string. \
            For example, \'0.1,0,0,0\' denotes setting use_activerl to 0.1 and use_rbc, use_random, and use_random_reset to 0",
        default=None
    )

def define_constants(args):
    global CL_FOLDER, CL_EVAL_PATHS
    # "./data/single_building" if args.single_building_eval else "./data/all_buildings"
    CL_FOLDER = args.cl_eval_folder
    CL_EVAL_PATHS = [os.path.join(CL_FOLDER, "Test_cold_Texas/schema.json"), os.path.join(CL_FOLDER, "Test_dry_Cali/schema.json"),
                     os.path.join(CL_FOLDER, "Test_hot_new_york/schema.json"), os.path.join(CL_FOLDER, "Test_snowy_Cali_winter/schema.json")]


def process_combo_args(args):
    if args.sinergym_sweep is not None:
        arglist = args.sinergym_sweep.split(",")
        if len(arglist) != 4:
            raise ValueError(
                "sinergym sweep combo arg does not have four elements")
        args.use_activerl = float(arglist[0])
        args.use_rbc = int(arglist[1])
        args.use_random = int(arglist[2])
        args.use_random_reset = float(arglist[3])
    return args