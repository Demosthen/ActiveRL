from dm_control.locomotion.tasks.random_goal_maze import RepeatSingleGoalMazeAugmentedWithTargets, RepeatSingleGoalMaze, DEFAULT_CONTROL_TIMESTEP, DEFAULT_PHYSICS_TIMESTEP, DEFAULT_ALIVE_THRESHOLD
from dm_control.locomotion.props import target_sphere
from dm_control.composer.observation import observable as observable_lib
import labmaze
import numpy as np
from dm_control.locomotion.arenas import covering, mazes as maze_arenas

_WALL_GEOM_GROUP = 3

def str_maze_to_one_hot_maze(entity_layer):
    TOKENS = ['.', '+', labmaze.defaults.OBJECT_TOKEN, labmaze.defaults.SPAWN_TOKEN]
    TOKEN_DICT = {i: token for i, token in enumerate(TOKENS)}
    entity_ordinal = np.zeros(entity_layer.shape)
    out_shape = list(entity_layer.shape) + [len(TOKENS)]
    out = np.zeros(out_shape)
    for i, token in enumerate(TOKENS):
        out[entity_layer == token][:, i] = 1
    return out 

class DM_Maze_Arena(maze_arenas.MazeWithTargets):
    def _make_wall_geoms(self, wall_char):
        i = 0
        for x in range(len(self._maze.entity_layer)):
            for y in range(len(self._maze.entity_layer[0])):
                wall_mid = covering.GridCoordinates(
                    (x + 0.5),
                    (y + 0.5))
                curr_char = self._maze.entity_layer[x, y]
                is_wall = int(curr_char == wall_char)
                wall_pos = np.array([(wall_mid.x - self._x_offset) * self._xy_scale,
                                    -(wall_mid.y - self._y_offset) * self._xy_scale,
                                    is_wall * self._z_height / 2 + 0.001])
                wall_size = np.array([self._xy_scale / 2,
                                        self._xy_scale / 2,
                                        is_wall * self._z_height / 2 + 0.001])
                self._maze_body.add('geom', name='wall{}_{}'.format(wall_char, i),
                                    type='box', pos=wall_pos, size=wall_size,
                                    group=_WALL_GEOM_GROUP)
                self._make_wall_texturing_planes(wall_char, i, wall_pos, wall_size)
                i += 1

class DM_Maze_Task(RepeatSingleGoalMazeAugmentedWithTargets):
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
               control_timestep=DEFAULT_CONTROL_TIMESTEP,
               enable_global_task_observables=False):
    super(RepeatSingleGoalMazeAugmentedWithTargets, self).__init__(
        walker=walker,
        target=main_target,
        maze_arena=maze_arena,
        target_reward_scale=target_reward_scale,
        randomize_spawn_position=randomize_spawn_position,
        randomize_spawn_rotation=randomize_spawn_rotation,
        rotation_bias_factor=rotation_bias_factor,
        aliveness_reward=aliveness_reward,
        aliveness_threshold=aliveness_threshold,
        contact_termination=contact_termination,
        physics_timestep=physics_timestep,
        control_timestep=control_timestep,
        enable_global_task_observables=enable_global_task_observables)
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

    if enable_global_task_observables:
    # Reveal maze text map as observable.
        maze_obs = observable_lib.Generic(
            lambda _: str_maze_to_one_hot_maze(self._maze_arena.maze.entity_layer))
        maze_obs.enabled = True
        old_str_maze_obs = self._task_observables['maze_layout']
        old_str_maze_obs.enabled = False
        self._task_observables['maze_layout'] = maze_obs


