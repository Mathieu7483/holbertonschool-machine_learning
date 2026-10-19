#!/usr/bin/env python3
"""Write a function that performs the Monte Carlo method
for estimating the value of a state in an environment."""
import numpy as np
import gymnasium as gym


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
    for _ in range(episodes):
        state, _ = env.reset()
        states = []
        rewards = []
        done = False

        for _ in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states.append(state)
            rewards.append(reward)

            if done:
                break
            state = next_state

        G = 0
        for t in reversed(range(len(states))):
            G = gamma * G + rewards[t]
            V[states[t]] += alpha * (G - V[states[t]])

    return V
