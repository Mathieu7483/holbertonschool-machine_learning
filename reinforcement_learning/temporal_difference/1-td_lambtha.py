#!/usr/bin/env python3
"""Write the function that performs the TD(λ) algorithm"""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """Function that performs the TD(λ) algorithm.

    Args:
        env: The OpenAI Gym environment instance.
        V: A numpy.ndarray of shape (s,) containing
        the value estimate for each state.
        policy: A function that takes in a state and
        returns the next action to take.
        lambtha: The eligibility trace factor.
        episodes: The total number of episodes to train over.
        max_steps: The maximum number of steps per episode.
        alpha: The learning rate.
        gamma: The discount rate.

    Returns:
        V: The updated value estimate for each state.
    """
    for episode in range(episodes):
        state = env.reset()[0]
        eligibility_trace = np.zeros_like(V)

        for step in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)

            delta = reward + gamma * V[next_state] - V[state]
            eligibility_trace[state] += 1

            V += alpha * delta * eligibility_trace
            eligibility_trace *= gamma * lambtha

            if terminated or truncated:
                break

            state = next_state

    return V
