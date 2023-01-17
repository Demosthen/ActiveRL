"""Implementation of custom reward"""

from datetime import datetime
from math import exp
from typing import Dict, List, Tuple, Union, Any
from gym import Env
from sinergym.utils.rewards import BaseReward, LinearReward

YEAR = 2021


class FangerReward(LinearReward):

    def __init__(
        self,
        env: Env,
        ppd_variable: Union[str, list],
        occupancy_variable: Union[str, list],
        energy_variable: str,
        energy_weight: float = 0.5,
        lambda_energy: float = 1e-4,
        lambda_ppd: float = 1.0
    ):
        """
        Linear reward function using Fanger PPD thermal comfort.
        It considers the energy consumption and the PPD thermal comfort metric, as well as occupancy.
        .. math::
            R = - W * lambda_E * power - (1 - W) * lambda_T * Fanger_metric * (occupancy > 0)
        Args:
            env (Env): Gym environment.
            temperature_variable (Union[str, list]): Name(s) of the temperature variable(s).
            energy_variable (str): Name of the energy/power variable.
            energy_weight (float, optional): Weight given to the energy term. Defaults to 0.5.
            lambda_energy (float, optional): Constant for removing dimensions from power(1/W). Defaults to 1e-4.
            lambda_temperature (float, optional): Constant for removing dimensions from temperature(1/C). Defaults to 1.0.
        """

        super(LinearReward, self).__init__(env)

        # Name of the variables
        self.ppd_name = ppd_variable
        self.energy_name = energy_variable
        self.occupancy_name = occupancy_variable

        # Reward parameters
        self.W_energy = energy_weight
        self.lambda_energy = lambda_energy
        self.lambda_temp = lambda_ppd

    def _get_comfort(self,
                     obs_dict: Dict[str,
                                    Any]) -> Tuple[float,
                                                   List[float]]:
        """Calculate the comfort term of the reward.
        Returns:
            Tuple[float, List[float]]: comfort penalty and List with temperatures used.
        """

        ppds = [v for k, v in obs_dict.items() if k in self.ppd_name]
        occupancies = [v for k, v in obs_dict.items() if k in self.occupancy_name]
        comfort = 0.0
        for ppd, occupancy in zip(ppds, occupancies):
            zone_comfort = ppd * (occupancy > 0)
            # If ppd < 20% it is within ASHRAE standards so it is not penalized
            if zone_comfort >= 20:
                comfort += zone_comfort
        return comfort, (0, 0)