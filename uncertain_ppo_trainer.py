import warnings
from typing import Any, Dict, Optional, Type, Union, List

import numpy as np
import torch as th
import torch.nn as nn
from gym import spaces
from torch.nn import functional as F

from ray.rllib.algorithms.ppo import PPO

from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TrainerConfigDict

from uncertain_ppo import UncertainPPOTorchPolicy

class UncertainPPO(PPO):

    @override(PPO)
    def get_default_policy_class(self, config: TrainerConfigDict) -> Type[Policy]:
        if config["framework"] == "torch":

            return UncertainPPOTorchPolicy
        else:
            return NotImplementedError
