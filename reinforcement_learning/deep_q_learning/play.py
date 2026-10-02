#!/usr/bin/env python3
"""
Displays games of Atari Breakout played by the DQN agent trained
by train.py, using the weights saved in policy.h5.
"""
import gymnasium as gym
import numpy as np
from keras.optimizers.legacy import Adam
from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy

from train import INPUT_SHAPE, AtariProcessor, CompatibilityWrapper, \
    build_model


def main():
    """
    Loads policy.h5 and plays with a greedy policy. Since a lost life
    ends an episode in the wrapper, 15 episodes are about 5 games.
    """
    env = CompatibilityWrapper(
        gym.make('ALE/Breakout-v5', obs_type='grayscale',
                 render_mode='human'))
    nb_actions = env.action_space.n
    model = build_model(INPUT_SHAPE, nb_actions)
    dqn = DQNAgent(model=model,
                   nb_actions=nb_actions,
                   memory=SequentialMemory(limit=1000, window_length=4),
                   processor=AtariProcessor(),
                   policy=GreedyQPolicy(),
                   test_policy=GreedyQPolicy())
    dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])
    dqn.load_weights('policy.h5')

    history = dqn.test(env, nb_episodes=15, visualize=False)
    print('Mean reward per episode:',
          np.mean(history.history['episode_reward']))
    env.close()


if __name__ == '__main__':
    main()
