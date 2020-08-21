from gym.envs.registration import register

register(
    id='simglucose-v2',
    entry_point='simglucose.envs:T1DSimEnv',
)
