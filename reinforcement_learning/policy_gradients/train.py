#!/usr/bin/env python3
"""Module that implements a full training loop for policy gradient."""
import numpy as np

policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """Implements a full training loop using policy gradient.

    Args:
        env: initial environment
        nb_episodes (int): number of episodes used for training
        alpha (float): the learning rate
        gamma (float): the discount factor

    Returns:
        list: all values of the score (sum of all
        rewards during one episode loop)
    """
    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n

    weight = np.random.rand(n_states, n_actions)
    scores = []

    for episode in range(nb_episodes):
        state, _ = env.reset()
        episode_rewards = []
        episode_grads = []

        terminated = False
        truncated = False

        while not terminated and not truncated:
            action, grad = policy_gradient(state, weight)
            next_state, reward, terminated, truncated, _ = env.step(action)

            episode_rewards.append(reward)
            episode_grads.append(grad)
            state = next_state

        score = sum(episode_rewards)
        scores.append(score)

        T = len(episode_rewards)
        G = 0.0
        returns = [0.0] * T
        for t in reversed(range(T)):
            G = episode_rewards[t] + gamma * G
            returns[t] = G

        for t in range(T):
            weight += alpha * returns[t] * episode_grads[t]

        print(f"Episode: {episode} Score: {score}")

    return scores
