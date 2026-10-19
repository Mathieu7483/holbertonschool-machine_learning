#!/usr/bin/env python3
"""Write the function that performs SARSA(λ)"""
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """Function that performs SARSA(λ).

    Args:
        env: The OpenAI Gym environment instance.
        Q: A numpy.ndarray of shape (s, a) containing the Q-table.
        lambtha: The eligibility trace factor.
        episodes: The total number of episodes to train over.
        max_steps: The maximum number of steps per episode.
        alpha: The learning rate.
        gamma: The discount rate.
        epsilon: The initial threshold for epsilon greedy.
        min_epsilon: The minimum value that epsilon should decay to.
        epsilon_decay: The decay rate for updating epsilon between episodes.

    Returns:
        Q: The updated Q-table.
    """
    for episode in range(episodes):
        state = env.reset()[0]
        eligibility_trace = np.zeros_like(Q)

        if np.random.uniform() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        for step in range(max_steps):
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if np.random.uniform() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(Q[next_state])

            if done:
                delta = reward - Q[state][action]
            else:
                delta = (reward + gamma * Q[next_state][next_action]
                         - Q[state][action])

            eligibility_trace[state][action] += 1

            Q += alpha * delta * eligibility_trace
            eligibility_trace *= gamma * lambtha

            if done:
                break

            state, action = next_state, next_action

        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))

    return Q
