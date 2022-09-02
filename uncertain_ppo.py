import warnings
from typing import Any, Dict, Optional, Type, Union, List

import numpy as np
import torch as th
import torch.nn as nn
from gym import spaces
from torch.nn import functional as F

from ray.rllib.algorithms.ppo import PPOTorchPolicy


class UncertainPPOTorchPolicy(PPOTorchPolicy):
    """PyTorch policy class used with PPO."""

    def __init__(self, observation_space, action_space, config):

        super().__init__(
            observation_space,
            action_space,
            config,
        )
        self.num_dropouts_evals = config["env_config"]["num_dropouts_evals"]
        print("uncertainppo")
    
    # def compute_uncertainty(self, obs_tensor: th.Tensor):
    #     """
    #         Computes the uncertainty of the neural network
    #         for this observation by running inference with different
    #         dropout masks and measuring the variance of the 
    #         critic network's output

    #         :param obs_tensor: torch tensor of observation(s) to compute
    #                 uncertainty for. Make sure it is on the same device
    #                 as the model
    #         :return: How uncertain the model is about the value for each
    #                 observation
    #     """
    #     values = []
    #     for i in range(self.num_dropout_evals):
    #         actions, vals, log_probs = self.policy(obs_tensor)
    #         values.append(vals)
    #     values = th.concat(values)
    #     uncertainty = th.std(values, dim=0)
    #     return uncertainty



