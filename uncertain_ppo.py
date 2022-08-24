import warnings
from typing import Any, Dict, Optional, Type, Union

import numpy as np
import torch as th
import torch.nn as nn
from gym import spaces
from torch.nn import functional as F

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from stable_baselines3.ppo.ppo import PPO
from stable_baselines3.common.utils import obs_as_tensor, safe_mean

class UncertainPPO(PPO):
    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        num_dropout_evals=10,
    ):

        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            target_kl=target_kl,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            _init_setup_model=_init_setup_model,
            create_eval_env=create_eval_env,
            seed=seed,
        )
        self.num_dropout_evals=num_dropout_evals
        #TODO: assert dropout is being used by the policy

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
        values = []
        for i in range(self.num_dropout_evals):
            actions, vals, log_probs = self.policy(obs_tensor)
            values.append(vals)
        values = th.concat(values)
        uncertainty = th.std(values, dim=0)
        return uncertainty


