import numpy as np
import torch as th
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.algorithms.ppo import PPOTorchPolicy
from ray.rllib.utils.torch_utils import apply_grad_clipping
import torch.nn as nn


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
        self.stop_gradient = False

    def get_value(self, **input_dict):
        input_dict = SampleBatch(input_dict)
        input_dict = self._lazy_tensor_dict(input_dict)
        model_out, _ = self.model(input_dict)
        return self.model.value_function()

    def get_action(self, **input_dict):
        input_dict = SampleBatch(input_dict)
        # input_dict = self._lazy_tensor_dict(input_dict)
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
        values = th.stack(values)
        uncertainty = th.var(values, dim=0)
        self.model.train(orig_mode)
        return uncertainty

    def compute_reward_uncertainty(
        self, obs_tensor: th.Tensor, next_obs_tensor: th.Tensor
    ):
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

    def extra_grad_process(self, local_optimizer, loss):
        if self.stop_gradient:
            return self.apply_stop_gradient(local_optimizer, loss)
        else:
            return apply_grad_clipping(self, local_optimizer, loss)

    def apply_stop_gradient(self, optimizer, loss):
        """Sets already computed grads inside `optimizer` to 0.

        Args:
            policy: The TorchPolicy, which calculated `loss`.
            optimizer: A local torch optimizer object.
            loss: The torch loss tensor.

        Returns:
            An info dict containing the "grad_norm" key and the resulting clipped
            gradients.
        """
        grad_gnorm = 0
        if self.config["grad_clip"] is not None:
            clip_value = self.config["grad_clip"]
        else:
            clip_value = np.inf

        for param_group in optimizer.param_groups:
            # Make sure we only pass params with grad != None into torch
            # clip_grad_norm_. Would fail otherwise.
            params = list(filter(lambda p: p.grad is not None, param_group["params"]))
            if params:
                # PyTorch clips gradients inplace and returns the norm before clipping
                # We therefore need to compute grad_gnorm further down (fixes #4965)
                for p in params:
                    p.grad.detach().mul_(0)
                global_norm = nn.utils.clip_grad_norm_(params, clip_value)

                if isinstance(global_norm, th.Tensor):
                    global_norm = global_norm.cpu().numpy()

                grad_gnorm += min(global_norm, clip_value)

        if grad_gnorm > 0:
            return {"grad_gnorm": grad_gnorm}
        else:
            # No grads available
            return {}
