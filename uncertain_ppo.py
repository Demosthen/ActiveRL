from ast import Call
from typing import Any, Callable, Dict, Optional, Type, Union, List

import numpy as np
import torch as th
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.algorithms.ppo import PPOTorchPolicy
from rbc_agent import RBCAgent

class UncertainPPOTorchPolicy(PPOTorchPolicy):
    """PyTorch policy class used with PPO."""

    def __init__(self, observation_space, action_space, config):

        super().__init__(
            observation_space,
            action_space,
            config,
        )
        print(self.model)
        self.num_dropout_evals = config["model"]["num_dropout_evals"]
        self.use_rbc_residual = config["cl_use_rbc_residual"]
        if self.use_rbc_residual:
            #Shrink initial weights to make sure initial actions are close to just passing rbc controller
            with th.no_grad():
                self._logits._model[0].weight /= 1000
                self._logits._model[0].bias /= 1000
    
    def get_value(self, **input_dict):
        input_dict = SampleBatch(input_dict)
        input_dict = self._lazy_tensor_dict(input_dict)
        model_out, _ = self.model(input_dict)
        return self.model.value_function()

    def get_action(self, **input_dict):
        input_dict = SampleBatch(input_dict)
        #input_dict = self._lazy_tensor_dict(input_dict)
        return self.compute_actions_from_input_dict(input_dict)[0]

    def compute_value_uncertainty(self, obs_tensor: th.Tensor):
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
        uncertainty = th.var(values, dim=0)
        self.model.train(orig_mode)
        return uncertainty

    def compute_reward_uncertainty(self, obs_tensor: th.Tensor, next_obs_tensor: th.Tensor):
        orig_mode = self.model.training
        self.model.train()
        rewards = []
        for _ in range(self.num_dropout_evals):
            curr_val = self.get_value(obs=obs_tensor, training=True)
            next_val = self.get_value(obs=next_obs_tensor, training=True)
            rew = curr_val - self.config["gamma"] * next_val
            rewards.append(rew)
        rewards = th.concat(rewards)
        # Since curr_val and next_val are sampled independently
        # the variance of their sum should be the sum of their variances
        # so divide by two so it's still comparable to the planning model's reward uncertainty
        uncertainty = th.var(rewards, dim=0) / 2
        self.model.train(orig_mode)
        return uncertainty

