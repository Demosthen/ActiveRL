# %%
import ctypes
import ctypes.util
ctypes.CDLL(ctypes.util.find_library('GL'), ctypes.RTLD_GLOBAL)

# %%
# The basic mujoco wrapper.
from dm_control import mujoco

# Access to enums and MuJoCo library functions.
from dm_control.mujoco.wrapper.mjbindings import enums
from dm_control.mujoco.wrapper.mjbindings import mjlib

# PyMJCF
from dm_control import mjcf

# Composer high level imports
from dm_control import composer
from dm_control.composer.observation import observable
from dm_control.composer import variation

# Imports for Composer tutorial example
from dm_control.composer.variation import distributions
from dm_control.composer.variation import noises
from dm_control.locomotion.arenas import floors

# Control Suite
from dm_control import suite

# Run through corridor example
from dm_control.locomotion.walkers import cmu_humanoid, ant
from dm_control.locomotion.arenas import corridors as corridor_arenas, mazes as maze_arenas
from dm_control.locomotion.tasks import corridors as corridor_tasks, RepeatSingleGoalMaze, RepeatSingleGoalMazeAugmentedWithTargets

# Soccer
from dm_control.locomotion import soccer

# Manipulation
from dm_control import manipulation
import labmaze
from dm_maze_wrappers.dm_wrapper import DM_Maze_Wrapper
from dm_maze_wrappers.dm_maze import DM_Maze_Task, DM_Maze_Arena, DM_Maze_Env
from core.utils import read_gridworld, grid_desc_to_dm

# %%
#@title Other imports and helper functions

# General
import copy
import os
import itertools
from IPython.display import clear_output
import numpy as np

# Graphics-related
import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from IPython.display import HTML
import PIL.Image
# Internal loading of video libraries.

# Use svg backend for figure rendering
%config InlineBackend.figure_format = 'svg'

# Font sizes
SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Inline video helper function
if os.environ.get('COLAB_NOTEBOOK_TEST', False):
  # We skip video generation during tests, as it is quite expensive.
  display_video = lambda *args, **kwargs: None
else:
  def display_video(frames, framerate=30):
    height, width, _ = frames[0].shape
    dpi = 70
    orig_backend = matplotlib.get_backend()
    matplotlib.use('Agg')  # Switch to headless 'Agg' to inhibit figure rendering.
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    matplotlib.use(orig_backend)  # Switch back to the original backend.
    ax.set_axis_off()
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])
    def update(frame):
      im.set_data(frame)
      return [im]
    interval = 1000/framerate
    anim = animation.FuncAnimation(fig=fig, func=update, frames=frames,
                                   interval=interval, blit=True, repeat=False)
    return HTML(anim.to_html5_video())

# Seed numpy's global RNG so that cell outputs are deterministic. We also try to
# use RandomState instances that are local to a single cell wherever possible.
np.random.seed(42)

# %%
#@title A static model {vertical-output: true}

static_model = """
<mujoco>
  <worldbody>
    <light name="top" pos="0 0 1"/>
    <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
    <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
  </worldbody>
</mujoco>
"""
physics = mujoco.Physics.from_xml_string(static_model)
pixels = physics.render()
PIL.Image.fromarray(pixels)

# %%
labmaze.defaults.__dict__.keys()

# %%
walker = ant.Ant()
grid_desc, rew_map, wind_p = read_gridworld("gridworlds/bad_bubble.txt")
print(grid_desc)
maze_str, subtarget_rews = grid_desc_to_dm(grid_desc, rew_map, wind_p)
print(maze_str)
# maze_str = "**********\n*........*\n*..G..G..*\n*....P...*\n*........*\n**********\n"
# subtarget_rews = [10, -10]
arena = DM_Maze_Arena(maze=labmaze.FixedMazeWithRandomGoals(maze_str))
print(arena._spawn_positions)
task = DM_Maze_Task(walker, None, arena, 1, enable_global_task_observables=True)

# %%
config = {
  "maze_str": maze_str, 
  "aliveness_reward": 0,
  "distance_reward_scale": 0,
  "subtarget_rews": subtarget_rews,
  "random_state": np.random.RandomState(42),
  "strip_singleton_obs_buffer_dim": True,
  "time_limit": 10,
  "use_all_geoms": True,
  "walker": "ball"
}
env = DM_Maze_Wrapper(config)
obs = env.reset({"absolute_position": [-1, -3, 0]})
#print({k: o.shape for k, o in obs.items()})
#print(obs["absolute_position"])
imgs = []
for i in range(1):
  observation, reward, done, info = env.step(env.action_space.sample())
  print(reward)
  pixels = []
  for camera_id in range(3):
    pixels.append(env.physics.render(camera_id=camera_id, width=240))
  display(PIL.Image.fromarray(np.hstack(pixels)))

# %%


# %%
env._task._subtargets

# %%
[subtarget.activated for subtarget in env._task._subtargets]
env._task._subtarget_rews

# %%
env._task._maze_arena.target_positions

# %%



