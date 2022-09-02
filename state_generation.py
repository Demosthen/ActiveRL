from typing import Callable, Tuple
from uncertain_ppo import UncertainPPOTorchPolicy
import torch
from gym.spaces import Space, Box, Discrete
import torch.optim as optim
import cooper
import numpy as np

class BoundedUncertaintyMaximization(cooper.ConstrainedMinimizationProblem):
    def __init__(self, lower_bounds, upper_bounds, lower_bounded_idxs, upper_bounded_idxs, agent):
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.lower_bounded_idxs = lower_bounded_idxs
        self.upper_bounded_idxs = upper_bounded_idxs
        self.agent = agent
        super().__init__(is_constrained=True)

    def closure(self, obs):

        # Negative sign added since we want to *maximize* the entropy
        uncertainty = - self.agent.compute_uncertainty(obs)

        # Entries of p >= 0 (equiv. -p <= 0)
        ineq_defect = (obs[self.lower_bounded_idxs] - self.lower_bounds) + (self.upper_bounds - obs[self.upper_bounded_idxs])

        return cooper.CMPState(loss=uncertainty, ineq_defect=ineq_defect)

def get_space_bounds(obs_space: Space):
    if isinstance(obs_space, Box):
        return obs_space.low, obs_space.high
    elif isinstance(obs_space, Discrete):
        return np.atleast_1d(obs_space.start), np.atleast_1d(obs_space.start + obs_space.n)
    else:
        raise NotImplementedError

# def generate_states(agent: UncertainPPO, obs_space: Space, num_descent_steps: int = 10, batch_size: int = 64, projection_fn: Callable = lambda x: x, use_coop=True):
#     """
#         Generates states by doing gradient descent to increase an agent's uncertainty
#         on states starting from random noise

# #         :param agent: the agent 
# #         :param obs_shape: the shape of a single observation
# #         :param num_descent_steps: the number of gradient descent steps to do
# #         :param batch_size: the number of observations to concurrently process
# #         :param projection_fn: a function to project a continuous, unconstrained observation vector
# #                                 to the actual observation space (e.g. if the observation space is actually
# #                                 discrete then you can round the features in the observation vector)
# #     """
# #     #TODO: make everything work with batches
# #     agent.policy.train()
#     for param in agent.policy.parameters():
#         param.requires_grad = True
#     lower_bounds, upper_bounds = get_space_bounds(obs_space)
#     lower_bounded_idxs = np.logical_not(np.isinf(lower_bounds))
#     upper_bounded_idxs = np.logical_not(np.isinf(upper_bounds))
#     print(lower_bounds, upper_bounds)
#     obs = []
#     with torch.no_grad():
#         for i in range(batch_size):
#             random_obs = obs_space.sample()
#             obs.append(torch.tensor(random_obs, device = agent.device, dtype=torch.float32))
#     obs = torch.nn.Parameter(torch.stack(obs), requires_grad=True)
#     #projected_obs = projection_fn(obs)
#     if use_coop:
#         cmp = BoundedUncertaintyMaximization(
#                                                 torch.tensor(lower_bounds[lower_bounded_idxs], device=agent.device), 
#                                                 torch.tensor(upper_bounds[upper_bounded_idxs], device=agent.device), 
#                                                 torch.tensor(lower_bounded_idxs, device=agent.device), 
#                                                 torch.tensor(upper_bounded_idxs, device=agent.device), 
#                                                 agent)
#         formulation = cooper.LagrangianFormulation(cmp)

#         primal_optimizer = cooper.optim.ExtraAdam([obs])

#         # Define the dual optimizer. Note that this optimizer has NOT been fully instantiated
#         # yet. Cooper takes care of this, once it has initialized the formulation state.
#         dual_optimizer = cooper.optim.partial_optimizer(cooper.optim.ExtraAdam)

# #         # Wrap the formulation and both optimizers inside a ConstrainedOptimizer
#         optimizer = cooper.ConstrainedOptimizer(formulation, primal_optimizer, dual_optimizer)
#     else:
#         optimizer = optim.Adam([obs])
#     uncertainties = []
    
#     for _ in range(num_descent_steps):
#         print(obs,_)
#         optimizer.zero_grad()
#         agent.policy.zero_grad()
#         uncertainty = agent.compute_uncertainty(obs)
#         if use_coop:
#             lagrangian = formulation.composite_objective(cmp.closure, obs)
#             formulation.custom_backward(lagrangian)
#             optimizer.step(cmp.closure, obs)
#         else:
#             loss = - uncertainty.sum()
#             loss.backward()
#             #print(obs.grad, next(agent.policy..parameters()).grad)
#             optimizer.step()
#         uncertainties.append(uncertainty)
#     projected_obs = projection_fn(obs)
#     #print(projected_obs)
#     return projected_obs, uncertainties
