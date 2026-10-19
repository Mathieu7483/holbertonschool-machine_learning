#!/usr/bin/env python3
"""Write the function that performs SARSA(λ)"""
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs SARSA(λ) (with eligibility traces) to estimate a Q-table.

    Parameters:
        env: the environment instance
        Q (numpy.ndarray): the Q-table, shape (s, a)
        lambtha (float): the eligibility trace factor
        episodes (int): total number of episodes to train over
        max_steps (int): maximum number of steps per episode
        alpha (float): the learning rate
        gamma (float): the discount rate
        epsilon (float): initial threshold for epsilon-greedy
        min_epsilon (float): minimum value epsilon decays to
        epsilon_decay (float): decay rate for epsilon between episodes

    Returns:
        numpy.ndarray: the updated Q-table
    """
    initial_epsilon = epsilon

    for episode in range(episodes):
        state = env.reset()[0]

        # Epsilon-greedy action selection for the first action
        if np.random.uniform(0, 1) > epsilon:

            action = np.argmax(Q[state, :])
        else:
            action = np.random.randint(0, Q.shape[1])

        # Init eligibility traces to zero
        eligibility_traces = np.zeros_like(Q)

        for _ in range(max_steps):
            new_state, reward, terminated, truncated, _ = env.step(action)

            # Epsilon-greedy action selection for the next action
            if np.random.uniform(0, 1) > epsilon:
                new_action = np.argmax(Q[new_state, :])
            else:
                new_action = np.random.randint(0, Q.shape[1])

            # TD error (no special case for terminal states)
            delta = (reward + gamma * Q[new_state, new_action]
                     - Q[state, action])

            eligibility_traces[state, action] += 1
            Q += alpha * delta * eligibility_traces
            eligibility_traces *= lambtha * gamma

            state = new_state
            action = new_action

            if terminated or truncated:
                break

        # Exploration rate decay
        epsilon = (min_epsilon + (initial_epsilon - min_epsilon) *

                   np.exp(-epsilon_decay * episode))

    return Q
