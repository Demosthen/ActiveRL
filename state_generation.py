from copy import deepcopy
from typing import Callable, Tuple
import torch
from gym.spaces import Space, Box, Discrete, Dict
import torch.optim as optim
import cooper
import numpy as np
from reward_predictor import RewardPredictor
from uncertain_ppo import UncertainPPOTorchPolicy
from uncertain_ppo_trainer import UncertainPPO
from citylearn_model_training.planning_model import LitPlanningModel
from resettable_env import ResettableEnv

class BoundedUncertaintyMaximization(cooper.ConstrainedMinimizationProblem):
    def __init__(self, obs, env: ResettableEnv, lower_bounds, upper_bounds, lower_bounded_idxs, upper_bounded_idxs, agent: UncertainPPOTorchPolicy, planning_model: LitPlanningModel=None, reward_model: RewardPredictor = None, planning_uncertainty_weight=1):
        self.env = env
        self.obs = obs
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.lower_bounded_idxs = lower_bounded_idxs
        self.upper_bounded_idxs = upper_bounded_idxs
        self.agent = agent
        self.planning_model = planning_model
        self.reward_model = reward_model
        self.planning_uncertainty_weight = planning_uncertainty_weight
        super().__init__(is_constrained=True)

    def closure(self, resettable):
        obs = self.env.combine_resettable_part(self.obs, resettable)
        
        if self.planning_model is None:
            # Negative sign added since we want to *maximize* the uncertainty
            loss = - self.agent.compute_value_uncertainty(obs).sum()
        else:
            action = self.agent.get_action(obs=obs)
            planning_uncertainty, next_obs = self.planning_model.compute_reward_uncertainty(obs, action, return_avg_state=True)
            agent_uncertainty = self.reward_model.compute_uncertainty(obs)
            denom = 1 + self.planning_uncertainty_weight
            loss = (- agent_uncertainty + self.planning_uncertainty_weight * planning_uncertainty) / denom
            #print("IS THIS LOSS? ", loss, loss.shape)

        # Entries of p >= 0 (equiv. -p <= 0)
        #resettable, obs = self.env.separate_resettable_part(obs)
        ineq_defect = torch.cat([resettable[self.lower_bounded_idxs] - self.lower_bounds, self.upper_bounds - resettable[self.upper_bounded_idxs]])
        return cooper.CMPState(loss=loss, ineq_defect=ineq_defect)

def get_space_bounds(obs_space: Space):
    if isinstance(obs_space, Box):
        return obs_space.low, obs_space.high
    elif isinstance(obs_space, Discrete):
        return np.atleast_1d(obs_space.start), np.atleast_1d(obs_space.start + obs_space.n)
    else:
        raise NotImplementedError

def sample_obs(env: ResettableEnv, batch_size: int, device):
    obs_space = env.observation_space
    if isinstance(obs_space, Dict):
        obs = {}
        with torch.no_grad():
            for _ in range(batch_size):
                random_obs = env.sample_obs()
                if isinstance(random_obs, dict):
                    for key, val in random_obs.items():
                        if key not in obs:
                            obs[key] = []
                        obs[key].append(torch.tensor(val, device = device, dtype=torch.float32, requires_grad=False))
        obs = {k: torch.stack(v) for k, v in obs.items()}
    else:
        obs = []
        with torch.no_grad():
            for i in range(batch_size):
                random_obs = env.sample_obs()
                if isinstance(random_obs, dict):
                    obs.append(torch.torch.tensor(random_obs, device = device, dtype=torch.float32, requires_grad=False))

        obs = torch.stack(obs)
    resettable_part, obs = env.separate_resettable_part(obs)
    
    resettable_part = torch.nn.Parameter(resettable_part.detach(), requires_grad=True) # Make a leaf tensor that is an optimizable Parameter
    obs = env.combine_resettable_part(obs, resettable_part)
    return obs, resettable_part

def generate_states(agent: UncertainPPOTorchPolicy, env: ResettableEnv, obs_space: Space, num_descent_steps: int = 10, batch_size: int = 1, no_coop=False, planning_model=None, reward_model=None, planning_uncertainty_weight=1):
    """
        Generates states by doing gradient descent to increase an agent's uncertainty
        on states starting from random noise

        :param agent: the agent 
        :param env: an environment that implements the separate_resettable_part and combine_resettable_part methods
        :param obs_space: the observation space
        :param num_descent_steps: the number of gradient descent steps to do
        :param batch_size: the number of observations to concurrently process (CURRENTLY DOESN'T DO ANYTHING, JUST SET IT TO 1)
        :param no_coop: whether or not to use the constrained optimization solver coop to make sure we don't go out of bounds. WILL LIKELY FAIL IF NOT SET TO TRUE
        :param planning_model: the planning model that was trained offline
        :param reward_model: the reward model you are training online
        :param projection_fn: a function to project a continuous, unconstrained observation vector
                                to the actual observation space (e.g. if the observation space is actually
                                discrete then you can round the features in the observation vector)
        :param planning_uncertainty_weight: relative weight to give to the planning uncertainty compared to agent uncertainty
    """
#     #TODO: make everything work with batches
    lower_bounds, upper_bounds = env.resettable_bounds()#get_space_bounds(obs_space)
    lower_bounded_idxs = np.logical_not(np.isinf(lower_bounds))
    upper_bounded_idxs = np.logical_not(np.isinf(upper_bounds))

    obs, resettable = sample_obs(env, batch_size, agent.device)

    if not no_coop:
        cmp = BoundedUncertaintyMaximization(
                                                obs,
                                                env,
                                                torch.tensor(lower_bounds[lower_bounded_idxs], device=agent.device), 
                                                torch.tensor(upper_bounds[upper_bounded_idxs], device=agent.device), 
                                                torch.tensor(lower_bounded_idxs[None, :], device=agent.device), 
                                                torch.tensor(upper_bounded_idxs[None, :], device=agent.device), 
                                                agent,
                                                planning_model,
                                                reward_model,
                                                planning_uncertainty_weight
                                                )
        formulation = cooper.LagrangianFormulation(cmp)

        primal_optimizer = cooper.optim.ExtraAdam([resettable], lr=0.1)

        # Define the dual optimizer. Note that this optimizer has NOT been fully instantiated
        # yet. Cooper takes care of this, once it has initialized the formulation state.
        dual_optimizer = cooper.optim.partial_optimizer(cooper.optim.ExtraAdam, lr=0.1)

#         # Wrap the formulation and both optimizers inside a ConstrainedOptimizer
        optimizer = cooper.ConstrainedOptimizer(formulation, primal_optimizer, dual_optimizer)
    else:
        optimizer = optim.Adam([resettable], lr=0.1)
    uncertainties = []
    
    for _ in range(num_descent_steps):
        optimizer.zero_grad()
        agent.model.zero_grad()
        if not no_coop:
            lagrangian = formulation.composite_objective(cmp.closure, resettable)
            formulation.custom_backward(lagrangian)
            optimizer.step(cmp.closure, resettable)
            uncertainties.append(cmp.state.loss)
        else:
            obs = env.combine_resettable_part(obs, resettable)
            uncertainty = agent.compute_value_uncertainty(obs)
            loss = - uncertainty.sum()
            loss.backward()
            optimizer.step()
            uncertainties.append(uncertainty.detach().cpu().numpy())
    return obs, uncertainties
