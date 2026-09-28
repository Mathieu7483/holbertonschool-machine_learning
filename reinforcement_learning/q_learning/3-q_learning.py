#!/usr/bin/env python3
"""Write the function that performs Q-learning"""
import numpy as np


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1, gamma=0.99,
          epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """Performs Q-learning

    Args:
        env: The FrozenLakeEnv environment
        Q: The Q-table
        episodes: The total number of episodes to train over
        max_steps: The maximum number of steps per episode
        alpha: The learning rate
        gamma: The discount factor
        epsilon: The initial threshold for choosing a random action
        min_epsilon: The minimum value that epsilon should decay to
        epsilon_decay: The decay rate for updating the value of epsilon

    Returns:
        Q: The updated Q-table
        total_rewards: A list containing the rewards per episode
    """
    total_rewards = []
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        rewards = 0

        for step in range(max_steps):
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            Q[state][action] += alpha * (
                reward + gamma * np.max(Q[next_state]) - Q[state][action]
            )
            state = next_state
            rewards += reward

            if done:
                break

        total_rewards.append(rewards)
        epsilon = max(min_epsilon, epsilon * (1 - epsilon_decay))

    return Q, total_rewards
