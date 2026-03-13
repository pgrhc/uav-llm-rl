from gymnasium.envs.registration import register

register(
    id="RouteAgent-v0",
    entry_point="uav_route_planner.envs.route_env:RouteEnv",
    max_episode_steps=1000,
)

register(
    id="RouteCurriculumAgent-v0",
    entry_point="uav_route_planner.envs.route_curriculum_env:RouteCurriculumEnv",
    max_episode_steps=1000,
)
