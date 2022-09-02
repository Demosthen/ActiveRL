from typing import Any, Dict, Optional, Type, Union, List

import numpy as np
import torch as th
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.algorithms.ppo import PPOTorchPolicy

class UncertainPPOTorchPolicy(PPOTorchPolicy):
    """PyTorch policy class used with PPO."""

    def __init__(self, observation_space, action_space, config):

        super().__init__(
            observation_space,
            action_space,
            config,
        )
        print(self.model)
        self.num_dropout_evals = config["env_config"]["num_dropout_evals"]
    
    def get_value(self, **input_dict):
            input_dict = SampleBatch(input_dict)
            input_dict = self._lazy_tensor_dict(input_dict)
            model_out, _ = self.model(input_dict)
            return self.model.value_function()

    def compute_uncertainty(self, obs_tensor: th.Tensor):
        """
            Computes the uncertainty of the neural network
            for this observation by running inference with different
            dropout masks and measuring the variance of the 
            critic network's output

            :param obs_tensor: torch tensor of observation(s) to compute
                    uncertainty for. Make sure it is on the same device
                    as the model
            :return: How uncertain the model is about the value for each
                    observation
        """
        orig_mode = self.model.training
        self.model.train()
        values = []
        for _ in range(self.num_dropout_evals):
            vals = self.get_value(obs=obs_tensor, training=True)
            values.append(vals)
        values = th.concat(values)
        uncertainty = th.std(values, dim=0)
        print("uncertainty", uncertainty)
        self.model.train(orig_mode)
        return uncertainty



