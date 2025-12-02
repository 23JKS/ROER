from wrappers.absorbing_states import AbsorbingStatesWrapper
# DMCEnv is imported lazily to avoid requiring dm_env when not using DMC environments
from wrappers.episode_monitor import EpisodeMonitor
from wrappers.frame_stack import FrameStack
from wrappers.repeat_action import RepeatAction
from wrappers.rgb2gray import RGB2Gray
from wrappers.single_precision import SinglePrecision
from wrappers.sticky_actions import StickyActionEnv
from wrappers.take_key import TakeKey
