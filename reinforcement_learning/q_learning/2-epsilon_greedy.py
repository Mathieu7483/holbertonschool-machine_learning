#!/usr/bin/env python3
"""Write a function that uses epsilon-greedy to determine the next action"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Uses epsilon-greedy to determine the next action

    Args:
        Q: The Q-table
        state: The current state
        epsilon: The threshold for choosing a random action

    Returns:
        action: The next action to take
    """
    if np.random.uniform(0, 1) < epsilon:
        action = np.random.randint(Q.shape[1])
    else:
        action = np.argmax(Q[state])
    return action
