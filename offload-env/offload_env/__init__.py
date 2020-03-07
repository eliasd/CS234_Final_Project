from gym.envs.registration import register

register(
    id='offload-v0',
    entry_point='offload_env.envs:OffloadEnv',
)