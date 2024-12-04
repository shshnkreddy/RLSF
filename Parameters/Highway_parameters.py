import numpy as np

params = {
    "observation": {
        "type": "GrayscaleObservation",
        "observation_shape": (64, 64),
        "stack_size": 1,
        "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
        "scaling": 1.75,
    },
    "action": {
        "type": "ContinuousAction",
        "acceleration_range": [-1.0, 1.0],
        "steering_range": [-0.015, 0.015],
    },
    "lanes_count": 3,
    "vehicles_count": 20,
    "duration": 100,  # [s]
    "initial_spacing": 2,
    "ego_spacing": 1.5,
    "collision_reward": -1,  # The reward received when colliding with a vehicle.
    "on_road_reward": 0,  # The reward received when on road.
    "right_lane_reward": 0.1,  # The reward received when driving on the right-most lane.
    # "reward_speed_range": [20, 30],  # [m/s] The reward for high speed is mapped linearly from this range to [0, HighwayEnv.HIGH_SPEED_REWARD].
    "simulation_frequency": 5,  # [Hz]
    "policy_frequency": 1,  # [Hz]
    "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
    "screen_width": 600,  # [px]
    "screen_height": 150,  # [px]
    "centering_position": [0.3, 0.5],
    "scaling": 5.5,
    "show_trajectories": False,
    "disable_collision_checks": True
}