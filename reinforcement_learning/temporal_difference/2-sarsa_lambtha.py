#!/usr/bin/env python3
"""Write the function that performs SARSA(λ)"""
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs the SARSA(λ) algorithm (with eligibility traces) to estimate
    a Q-table.

    Parameters:
        env (gym.Env): The FrozenLake environment instance.
        Q (numpy.ndarray): The given Q-table.
        lambtha (float): The eligibility trace factor.
        episodes (int, optional): The total number of episodes to train over.
        max_steps (int, optional): The maximum number of steps per episode.
        alpha (float, optional): The learning rate. Default is 0.1.
        gamma (float, optional): The discount rate. Default is 0.99.
        epsilon (float, optional): The initial threshold for epsilon-greedy.
        min_epsilon (float, optional): The minimum value for epsilon.
        epsilon_decay (float, optional): The decay rate for
        epsilon per episode.

    Returns:
        numpy.ndarray: The updated Q-table after training.
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

        for steps in range(max_steps):
            new_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Epsilon-greedy action selection for the next action
            if np.random.uniform(0, 1) > epsilon:
                new_action = np.argmax(Q[new_state, :])
            else:
                new_action = np.random.randint(0, Q.shape[1])

            # TD Error calculation (handling terminal states)
            if done:
                delta = reward - Q[state, action]
            else:
                delta = (reward + (gamma * Q[new_state, new_action]) -
                         Q[state, action])

            # 1. Accumulate trace for current state-action pair
            eligibility_traces[state, action] += 1

            # 2. Update the Q-table using the current traces
            Q += alpha * delta * eligibility_traces

            # 3. Decay the eligibility traces for the next steps
            eligibility_traces *= lambtha * gamma

            state = new_state
            action = new_action

            if done:
                break

        # Exploration rate decay
        epsilon = (min_epsilon + (initial_epsilon - min_epsilon) *
                   np.exp(-epsilon_decay * episode))

    return Q
