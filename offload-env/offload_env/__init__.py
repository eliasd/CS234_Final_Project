from gym.envs.registration import register
from offload_env.envs.offload_env import OffloadEnv

register(
    id='offload-v0',
    entry_point='offload_env.envs:OffloadEnv',
)