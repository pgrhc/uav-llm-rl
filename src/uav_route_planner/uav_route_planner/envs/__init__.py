from gymnasium.envs.registration import register

# Truncation handled by env.MAX_EPISODE_STEPS; no TimeLimit wrapper to avoid duplication
register(
    id="RouteAgent-v0",
    entry_point="uav_route_planner.envs.route_env:RouteEnv",
)

register(
    id="RouteCurriculumAgent-v0",
    entry_point="uav_route_planner.envs.route_curriculum_env:RouteCurriculumEnv",
)

