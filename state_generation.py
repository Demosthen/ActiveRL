from typing import Callable, Tuple
from uncertain_ppo import UncertainPPOTorchPolicy
import torch
from gym.spaces.space import Space
import torch.optim as optim

# def generate_states(agent: UncertainPPO, obs_space: Space, num_descent_steps: int = 10, batch_size: int = 64, projection_fn: Callable = lambda x: x):
#     """
#         Generates states by doing gradient descent to increase an agent's uncertainty
#         on states starting from random noise

#         :param agent: the agent 
#         :param obs_shape: the shape of a single observation
#         :param num_descent_steps: the number of gradient descent steps to do
#         :param batch_size: the number of observations to concurrently process
#         :param projection_fn: a function to project a continuous, unconstrained observation vector
#                                 to the actual observation space (e.g. if the observation space is actually
#                                 discrete then you can round the features in the observation vector)
#     """
#     #TODO: make everything work with batches
#     obs_batch_shape = [batch_size] + list(obs_space.shape)
#     #obs_batch_shape = list(obs_shape)
#     obs = []
#     with torch.no_grad():
#         for i in range(batch_size):
#             random_obs = obs_space.sample()
#             obs.append(torch.tensor(random_obs, device = agent.device, dtype=torch.float32))
#     obs = torch.stack(obs)
    
#     projected_obs = projection_fn(obs)
#     optimizer = optim.Adam([obs])
    
#     for _ in range(num_descent_steps):
#         optimizer.zero_grad()
#         agent.policy.zero_grad()
#         uncertainty = agent.compute_uncertainty(projected_obs)
#         loss = - uncertainty.sum()
#         loss.backward()
#         optimizer.step()
#         projected_obs = projection_fn(obs)
#     print(projected_obs)
#     return projected_obs
