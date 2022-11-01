from dm_control.locomotion.tasks.random_goal_maze import RepeatSingleGoalMazeAugmentedWithTargets, RepeatSingleGoalMaze, DEFAULT_CONTROL_TIMESTEP, DEFAULT_PHYSICS_TIMESTEP, DEFAULT_ALIVE_THRESHOLD, _NUM_RAYS
from dm_control.mujoco.wrapper import mjbindings
from dm_control.locomotion.props import target_sphere
from dm_control.composer.observation import observable as observable_lib
from dm_control.composer import Environment, ObservationPadding
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
    """
        A Mujoco arena that takes in a fixed maze and encodes the maze as a bunch of blocks with height between 0 and 1
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_original_spawn = True
        self.regenerate()

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

    def _find_spawn_and_target_positions(self):
        if self.use_original_spawn:
            grid_positions = self.find_token_grid_positions([
                labmaze.defaults.OBJECT_TOKEN, labmaze.defaults.SPAWN_TOKEN])
            self._target_grid_positions = tuple(
                grid_positions[labmaze.defaults.OBJECT_TOKEN])
            self._spawn_grid_positions = tuple(
                grid_positions[labmaze.defaults.SPAWN_TOKEN])
            self._target_positions = tuple(
                self.grid_to_world_positions(self._target_grid_positions))
            self._spawn_positions = tuple(
                self.grid_to_world_positions(self._spawn_grid_positions))

    def set_spawn(self, pos):
        self._spawn_positions = (pos,)
        self._spawn_grid_positions = self.world_to_grid_positions([pos])
        self.use_original_spawn = False

    def reset_original_spawn(self):
        self.use_original_spawn = True

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
                randomize_spawn_position=False, # changed from original
                randomize_spawn_rotation=False, # changed from original
                rotation_bias_factor=0,
                aliveness_reward=0.0,
                aliveness_threshold=DEFAULT_ALIVE_THRESHOLD,
                contact_termination=True,
                physics_timestep=DEFAULT_PHYSICS_TIMESTEP,
                control_timestep=DEFAULT_CONTROL_TIMESTEP,
                enable_global_task_observables=False,
                use_map_layout=False):
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
        # No-op to get VS Code to recognize the typing
        self._maze_arena: DM_Maze_Arena = self._maze_arena

        #Set subtargets
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

        # Change DeepMind's weird string array of maze layout to a 3D array where the last dimension is one hot vectors
        if enable_global_task_observables:
            if use_map_layout:
                # Reveal maze text map as observable.
                maze_obs = observable_lib.Generic(
                    lambda _: str_maze_to_one_hot_maze(self._maze_arena.maze.entity_layer))
                maze_obs.enabled = True
                old_str_maze_obs = self._task_observables['maze_layout']
                old_str_maze_obs.enabled = False
                self._task_observables['maze_layout'] = maze_obs
            else:
                self._task_observables['maze_layout'].enabled = False

    def _respawn(self, physics, random_state):
        self._walker.reinitialize_pose(physics, random_state)

        if self._randomize_spawn_position:
            self._spawn_position = self._maze_arena.spawn_positions[
                random_state.randint(0, len(self._maze_arena.spawn_positions))]
        else:
            self._spawn_position = self._maze_arena.spawn_positions[0]

        if self._randomize_spawn_rotation:
            # Move walker up out of the way before raycasting.
            self._walker.shift_pose(physics, [0.0, 0.0, 100.0])

            distances = []
            geomid_out = np.array([-1], dtype=np.intc)
            for i in range(_NUM_RAYS):
                theta = 2 * np.pi * i / _NUM_RAYS
                pos = np.array([self._spawn_position[0], self._spawn_position[1], 0.1],
                            dtype=np.float64)
                vec = np.array([np.cos(theta), np.sin(theta), 0], dtype=np.float64)
                dist = mjbindings.mjlib.mj_ray(
                    physics.model.ptr, physics.data.ptr, pos, vec,
                    None, 1, -1, geomid_out)
                distances.append(dist)

            def remap_with_bias(x):
                """Remaps values [-1, 1] -> [-1, 1] with bias."""
                return np.tanh((1 + self._rotation_bias_factor) * np.arctanh(x))

            max_theta = 2 * np.pi * np.argmax(distances) / _NUM_RAYS
            rotation = max_theta + np.pi * (
                1 + remap_with_bias(random_state.uniform(-1, 1)))

            quat = [np.cos(rotation / 2), 0, 0, np.sin(rotation / 2)]
            # Move walker back down.
            self._walker.shift_pose(physics, [0.0, 0.0, -100.0])
        else:
            quat = None
            # pos = np.array(self._spawn_position,
            #                 dtype=np.float64)
            # vec = np.array([0, 0, -1], dtype=np.float64)
            # dist = mjbindings.mjlib.mj_ray(
            #     physics.model.ptr, physics.data.ptr, pos, vec,
            #     None, 1, -1, geomid_out)
        self._walker.shift_pose(
            physics, self._spawn_position,
            quat,
            rotate_velocity=True)

class DM_Maze_Env(Environment):
    """Environment that wraps the task"""
    def __init__(self, task: DM_Maze_Task, time_limit=..., random_state=None, n_sub_steps=None, raise_exception_on_physics_error=True, strip_singleton_obs_buffer_dim=False, max_reset_attempts=1, delayed_observation_padding=ObservationPadding.ZERO):
        super().__init__(task, time_limit, random_state, n_sub_steps, raise_exception_on_physics_error, strip_singleton_obs_buffer_dim, max_reset_attempts, delayed_observation_padding)

    def reset(self, initial_state=None):
        task: DM_Maze_Task = self.task
        arena = task._maze_arena
        # Reset to initial state if specified.
        # Note that only the position part of the state is used to reset the environment.
        if initial_state is not None:
            pos = initial_state["absolute_position"]
            arena.set_spawn(pos)
        else:
            arena.reset_original_spawn()

        return super().reset()

