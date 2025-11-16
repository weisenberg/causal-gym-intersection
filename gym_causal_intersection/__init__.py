from gymnasium.envs.registration import register

register(
    id='UrbanCausalIntersection-v0',
    entry_point='gym_causal_intersection.envs.causal_intersection_env:UrbanCausalIntersectionEnv',
)

register(
    id='UrbanCausalIntersectionExtended-v0',
    entry_point='gym_causal_intersection.envs.causal_intersection_extended_env:UrbanCausalIntersectionExtendedEnv',
)

