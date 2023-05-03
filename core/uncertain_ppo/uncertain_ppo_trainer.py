from typing import Type

from ray.rllib.algorithms.ppo import PPO

import logging
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import TrainerConfigDict


from core.uncertain_ppo.uncertain_ppo import UncertainPPOTorchPolicy

logger = logging.getLogger(__name__)

class UncertainPPO(PPO):

    @override(PPO)
    def get_default_policy_class(self, config: TrainerConfigDict) -> Type[Policy]:
        if config["framework"] == "torch":

            return UncertainPPOTorchPolicy
        else:
            return NotImplementedError