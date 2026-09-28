#!/usr/bin/env python3
"""Write a function that initializes the Q-table"""
import numpy as np
import gymnasium as gym


def q_init(env):
    """Initializes the Q-table

    Args:
        env: The FrozenLakeEnv environment

    Returns:
        Q: The initialized Q-table
    """
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    return Q
