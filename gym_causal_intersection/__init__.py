from gymnasium.envs.registration import register

register(
    id='UrbanCausalIntersection-v0',
    entry_point='gym_causal_intersection.envs:UrbanCausalIntersectionEnv',
)

register(
    id='UrbanCausalIntersectionExtended-v0',
    entry_point='gym_causal_intersection.envs:UrbanCausalIntersectionExtendedEnv',
)

register(
    id='SimpleCausalIntersection-v0',
    entry_point='gym_causal_intersection.envs.simple_causal_env:SimpleCausalIntersectionEnv',
)
