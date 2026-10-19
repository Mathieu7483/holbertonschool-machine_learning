#!/usr/bin/env python3
"""Write a function that performs the Monte Carlo method
for estimating the value of a state in an environment."""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """Function that performs the Monte Carlo method
    for estimating the value of a state in an environment.

    Args:
        env: The OpenAI Gym environment instance.
        V: A numpy.ndarray of shape (s,) containing the value
        estimate for each state.
        policy: A function that takes in a state and returns
        an action to take in that state.
        episodes: The total number of episodes to sample from
        the environment.
        max_steps: The maximum number of steps per episode.
        alpha: The learning rate.
        gamma: The discount factor.

    Returns:
        A numpy.ndarray of shape (s,) containing the updated
        value estimate for each state.
    """
    for episode in range(episodes):
        state = env.reset()[0]
        episode_data = []

        for step in range(max_steps):
            action = policy(state)

            next_state, reward, terminated, truncated, _ = env.step(
                action)

            episode_data.append((state, reward))

            if terminated or truncated:
                break

            state = next_state

        G = 0
        episode_data = np.array(episode_data, dtype=int)

        for state, reward in reversed(episode_data):
            G = reward + gamma * G

            if state not in episode_data[:episode, 0]:
                V[state] = V[state] + alpha * (G - V[state])

    return V
