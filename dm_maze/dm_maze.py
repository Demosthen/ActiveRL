from dm_control.locomotion.tasks.random_goal_maze import RepeatSingleGoalMazeAugmentedWithTargets, DEFAULT_CONTROL_TIMESTEP, DEFAULT_PHYSICS_TIMESTEP, DEFAULT_ALIVE_THRESHOLD
from dm_control.locomotion.props import target_sphere
from dm_control.composer.observation import observable as observable_lib
import labmaze
import numpy as np

def str_maze_to_one_hot_maze(entity_layer):
    TOKENS = ['.', '+', labmaze.defaults.OBJECT_TOKEN, labmaze.defaults.SPAWN_TOKEN]
    TOKEN_DICT = {i: token for i, token in enumerate(TOKENS)}
    entity_ordinal = np.zeros(entity_layer.shape)
    out_shape = list(entity_layer.shape) + len(TOKENS)
    out = np.zeros(out_shape)
    for i, token in enumerate(TOKENS):
        out[entity_layer == token][:,:, i] = 1
    return out 
    
    

class DM_Maze(RepeatSingleGoalMazeAugmentedWithTargets):
  """Augments the single goal maze with many lower reward targets."""

  def __init__(self,
               walker,
               main_target,
               maze_arena,
               num_subtargets=20,
               target_reward_scale=10.0,
               subtarget_reward_scale=1.0,
               subtarget_colors=((0, 0, 0.4), (0, 0, 0.7)),
               randomize_spawn_position=True,
               randomize_spawn_rotation=True,
               rotation_bias_factor=0,
               aliveness_reward=0.0,
               aliveness_threshold=DEFAULT_ALIVE_THRESHOLD,
               contact_termination=True,
               physics_timestep=DEFAULT_PHYSICS_TIMESTEP,
               control_timestep=DEFAULT_CONTROL_TIMESTEP):
    super().__init__(
        walker,
        main_target,
        maze_arena,
        num_subtargets,
        target_reward_scale,
        subtarget_reward_scale,
        subtarget_colors,
        randomize_spawn_position,
        randomize_spawn_rotation,
        rotation_bias_factor,
        aliveness_reward,
        aliveness_threshold,
        contact_termination,
        physics_timestep,
        control_timestep)
    self._subtarget_reward_scale = subtarget_reward_scale
    self._subtargets = []
    for i in range(num_subtargets):
      subtarget = target_sphere.TargetSphere(
          radius=0.4, rgb1=subtarget_colors[0], rgb2=subtarget_colors[1],
          name='subtarget_{}'.format(i)
      )
      self._subtargets.append(subtarget)
      self._maze_arena.attach(subtarget)
    self._subtarget_rewarded = None

    # Reveal maze text map as observable.
    maze_obs = observable_lib.Generic(
        lambda _: self._maze_arena.maze.entity_layer)
    maze_obs.enabled = True