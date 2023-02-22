from copy import deepcopy
from typing import Dict, Optional, Union
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import PolicyID
from ray.rllib.policy import Policy
from ray.rllib.evaluation import RolloutWorker
from core.uncertain_ppo.uncertain_ppo_trainer import UncertainPPO
import numpy as np
from core.constants import *
from core.utils import flatten_dict_of_lists
from dm_maze_wrappers.viz_utils import draw_box
from core.callbacks import ActiveRLCallback
class DMMazeCallback(ActiveRLCallback):

    def __init__(self, num_descent_steps: int = 10, batch_size: int = 64, no_coop: bool = False, planning_model=None, config={}, run_active_rl=False, planning_uncertainty_weight=1, device="cpu", args={}, uniform_reset=False):
        super().__init__(num_descent_steps, batch_size, no_coop, planning_model, config, run_active_rl, planning_uncertainty_weight, device, args, uniform_reset)
        self.cell_index = -1
        self.num_cells = -1
        self.eval_rewards = []
        self.goal_reached = []
        self.num_envs = config["num_envs_per_worker"]
        self.env_to_cell_index = {k: -1 for k in range(self.num_envs)}

    def _get_img_bounds(self, img):
        """ Get bounds of foreground of image (assuming background is uniform and starts at edge of image)"""
        bg_value = img.flatten()[0]
        img = deepcopy(img)
        foreground_prev = np.nonzero(img)
        img[np.abs(img - bg_value) < 10] = 0
        foreground = np.nonzero(img)
        x_low, x_high = np.min(foreground[-2]), np.max(foreground[-2]) + 1
        y_low, y_high = np.min(foreground[-1]), np.max(foreground[-1]) + 1
        return x_low, x_high, y_low, y_high

    def _generate_highlighted_img(self, img: np.ndarray, per_cell_rewards: Dict, shift: float, scale: float):
        """Generates an image overlaid with a grid highlighted according to per_cell_rewards.
            WARNING: this is currently in-place."""
        x_low, x_high, y_low, y_high = self._get_img_bounds(img[:, :, :, :, :img.shape[-2]])
        highlight_img_h = x_high - x_low
        highlight_img_w = y_high - y_low
        highlight_img = np.zeros([3, highlight_img_h, highlight_img_w])
        for idx, reward in per_cell_rewards.items():
            grid_pos = self.grid_positions[int(idx)]
            scaled_reward = (reward + shift) * scale#(reward + 10000) / 300
            draw_box(highlight_img, grid_pos, self.grid_h, self.grid_w, scaled_reward)

        img[..., :, x_low:x_high, y_low:y_high] = img[..., :, x_low:x_high, y_low:y_high] * 0.75 + 0.25 * np.uint8(highlight_img)
        return img

    def on_evaluate_end(
        self, 
        *,
        algorithm: UncertainPPO,
        evaluation_metrics: dict,
        **kwargs)-> None:
        """
        Runs at the end of Algorithm.evaluate().
        """
        def access_eval_metrics(worker):
            if hasattr(worker, "callbacks") and worker.callbacks.num_cells > 0:
                worker.callbacks.is_evaluating = False
                return worker.callbacks.eval_rewards, worker.callbacks.goal_reached, worker.callbacks.grid_h, worker.callbacks.grid_w, worker.callbacks.grid_positions, worker.callbacks.world_positions, worker.callbacks.num_cells
            else:
                return []
        workers_out = [out for out in algorithm.evaluation_workers.foreach_worker(access_eval_metrics) if len(out) > 0]
        _, _, self.grid_h, self.grid_w, self.grid_positions, \
            self.world_positions, self.num_cells = workers_out[0]
        rewards = np.array([output[0] for output in workers_out])
        goal_reached = np.array([output[1] for output in workers_out])
        goal_reached_vid = evaluation_metrics["evaluation"]["episode_media"]["img"][0]
        goal_reached_img = goal_reached_vid[-1][None, None, :, :, :]
        evaluation_metrics["evaluation"]["single_img"] = goal_reached_img
        evaluation_metrics["evaluation"]["vid"] = goal_reached_vid[None, :, :, :, :]
        if self.full_eval_mode:
            rewards = np.mean(rewards, axis=0)
            per_cell_rewards = {f"{cell}": rew for cell, rew in enumerate(rewards)}
            goal_reached = np.mean(goal_reached, axis=0)
            per_cell_goal_reached = {f"{cell}": reached for cell, reached in enumerate(goal_reached)}
            evaluation_metrics["evaluation"]["per_cell_rewards"] = per_cell_rewards
            evaluation_metrics["evaluation"]["per_cell_goal_reached"] = per_cell_goal_reached
            shift = 0
            scale = 1
            goal_reached_img = self._generate_highlighted_img(deepcopy(goal_reached_img), per_cell_goal_reached, shift, scale)
            evaluation_metrics["evaluation"]["per_cell_goal_reached_img"] = goal_reached_img

    def on_episode_start(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Union[Episode, EpisodeV2],
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Callback run on the rollout worker before each episode starts.

        Args:
            worker: Reference to the current rollout worker.
            base_env: BaseEnv running the episode. The underlying
                sub environment objects can be retrieved by calling
                `base_env.get_sub_environments()`.
            policies: Mapping of policy id to policy objects. In single
                agent mode there will only be a single "default" policy.
            episode: Episode object which contains the episode's
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
            env_index: The index of the sub-environment that started the episode
                (within the vector of sub-environments of the BaseEnv).
            kwargs: Forward compatibility placeholder.
        """
        env = base_env.get_sub_environments()[env_index]
        # Get the single "default policy"
        policy = next(policies.values())
        episode.media["img"] = []
        episode.user_data["env_num_steps"] = 0
        needs_initialization = self.num_cells < 0
        
        if needs_initialization:
            # Initializes stuff needed for logging, that needs to know the arena's layout.
            arena = env.get_task()._maze_arena
            self.grid_h, self.grid_w = arena._maze.entity_layer.shape
            self.grid_positions = flatten_dict_of_lists(arena.find_token_grid_positions(RESPAWNABLE_TOKENS))
            self.world_positions = arena.grid_to_world_positions(self.grid_positions)
            self.num_cells = len(self.world_positions)
            self.eval_rewards = [0 for _ in range(self.num_cells)]
            self.goal_reached = [0 for _ in range(self.num_cells)]
            

        run_active_rl = np.random.random() < self.run_active_rl
        if self.is_evaluating and self.full_eval_mode:
            # Resets environment to all states, one by one.
            self.cell_index += 1
            self.env_to_cell_index[env_index] = self.cell_index
            initial_world_position = self.world_positions[self.cell_index % self.num_cells]
            initial_state = env.observation_space.sample()
            initial_state = env.combine_resettable_part(initial_state, initial_world_position)
            print(self.cell_index)
            env.reset(initial_state=initial_state)
        elif not self.is_evaluating and (run_active_rl or self.uniform_reset):
            # Actually run Active RL and reset the environment
            new_states = self.reset_env(policy, env, episode)
            if ACTIVE_STATE_VISITATION_KEY not in episode.custom_metrics:
                episode.hist_data[ACTIVE_STATE_VISITATION_KEY] = []
            to_log, _ = env.separate_resettable_part(new_states)
            to_log = np.array(to_log)
            to_log = to_log[0] * self.grid_w + to_log[1]
            episode.hist_data[ACTIVE_STATE_VISITATION_KEY].append(to_log)

    def on_episode_end(
        self,
        *,
        worker: RolloutWorker,
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Union[Episode, EpisodeV2],
        env_index: Optional[int] = None,
        **kwargs)-> None:
        """
        Runs when an episode is done.

        Args:
            worker: Reference to the current rollout worker.
            base_env : BaseEnv running the episode. The underlying sub environment
                objects can be retrieved by calling base_env.get_sub_environments().
            policies : Mapping of policy id to policy objects. In single agent mode
                there will only be a single “default_policy”.
            episode : Episode object which contains episode state. You can use the
                episode.user_data dict to store temporary data, and episode.custom_metrics
                to store custom metrics for the episode. In case of environment failures,
                episode may also be an Exception that gets thrown from the environment
                before the episode finishes. Users of this callback may then handle
                these error cases properly with their custom logics.
            kwargs : Forward compatibility placeholder.
        """
        env = base_env.get_sub_environments()[env_index]
        
        episode.custom_metrics["reached_goal"] = float(env.reached_goal_last_ep)
        if self.is_evaluating:
            if self.full_eval_mode:
                num_repeats_per_cell = self.config["evaluation_duration"]
                cell_index = self.env_to_cell_index[env_index]
                self.eval_rewards[cell_index % self.num_cells] += episode.total_reward / num_repeats_per_cell
                self.goal_reached[cell_index % self.num_cells] += float(env.reached_goal_last_ep) / num_repeats_per_cell
            pix = env.render()
            pix = np.transpose(pix, [2, 0, 1])
            episode.media["img"] = np.stack(episode.media["img"])

    def on_episode_step(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        episode: Union[Episode, EpisodeV2],
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Runs on each episode step.

        Args:
            worker: Reference to the current rollout worker.
            base_env: BaseEnv running the episode. The underlying
                sub environment objects can be retrieved by calling
                `base_env.get_sub_environments()`.
            policies: Mapping of policy id to policy objects.
                In single agent mode there will only be a single
                "default_policy".
            episode: Episode object which contains episode
                state. You can use the `episode.user_data` dict to store
                temporary data, and `episode.custom_metrics` to store custom
                metrics for the episode.
            env_index: The index of the sub-environment that stepped the episode
                (within the vector of sub-environments of the BaseEnv).
            kwargs: Forward compatibility placeholder.
        """
        env = base_env.get_sub_environments()[env_index]
        if episode.user_data["env_num_steps"] % 20 == 0:
            pix = env.render()
            pix = np.transpose(pix, [2, 0, 1])
            episode.media["img"].append(pix)
        episode.user_data["env_num_steps"] += 1