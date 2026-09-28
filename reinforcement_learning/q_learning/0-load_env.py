#!/usr/bin/env python3
"""Write a function that loads the pre-made FrozenLakeEnv
environment from gymnasium"""
import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """Loads the pre-made FrozenLakeEnv environment from gymnasium

    Args:
        desc (list): A list of lists containing a custom description
        of the map to load for the environment
        map_name (str): The name of the pre-made map to load
        for the environment
        is_slippery (bool): A boolean to determine if the ice
        is slippery or not

    Returns:
        env: The loaded FrozenLakeEnv environment
    """
    env = gym.make('FrozenLake-v1', desc=desc, map_name=map_name,
                   is_slippery=is_slippery)
    return env
