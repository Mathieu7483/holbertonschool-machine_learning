#!/usr/bin/env python3
"""Write a function that has the trained agent play an episode"""
import numpy as np


def play(env, Q, max_steps=100):
    """Has the trained agent play an episode

    Args:
        env: The FrozenLakeEnv environment
        Q: The Q-table
        max_steps: The maximum number of steps to take

    Returns:
        total_rewards: The total rewards obtained during the episode
        rendered_outputs: A list of the rendered outputs of the environment
    """
    env.unwrapped.render_mode = 'ansi'

    state, _ = env.reset()
    total_rewards = 0
    rendered_outputs = [env.render()]

    for _ in range(max_steps):
        action = np.argmax(Q[state])
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        total_rewards += reward
        rendered_outputs.append(env.render())

        if done:
            break

        state = next_state

    return total_rewards, rendered_outputs
