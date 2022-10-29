from ray.rllib.policy import Policy
import numpy as np
class RBCAgent:
    """
    Basic Rule based agent adopted from official Citylearn Rule based agent
    https://github.com/intelligent-environments-lab/CityLearn/blob/6ee6396f016977968f88ab1bd163ceb045411fa2/citylearn/agents/rbc.py#L23
    """

    def __init__(self, action_space):
        # example parameter
        self.action_space = action_space

    def compute_action(self, observation):
        """
        Simple rule based policy based on day or night time
        """
        hour = observation[2] # Hour index is 2 for all observations
        
        action = 0.0 # Default value
        if 9 <= hour <= 21:
            # Daytime: release stored energy
            action = -0.08
        elif (1 <= hour <= 8) or (22 <= hour <= 24):
            # Early nightime: store DHW and/or cooling energy
            action = 0.091
        action = np.repeat(np.array(action), repeats=self.action_space.shape[-1])
        return action