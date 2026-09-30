#!/usr/bin/env python3
"""
DQN training script for Atari Breakout using Keras-RL.
"""
import gymnasium as gym
import cv2
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Permute
from keras.optimizers.legacy import Adam
from rl.agents.dqn import DQNAgent
from rl.callbacks import ModelIntervalCheckpoint
from rl.memory import SequentialMemory
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.processors import Processor

INPUT_SHAPE = (4, 84, 84)


class AtariProcessor(Processor):
    """
    Preprocesses observations, batches and rewards.
    Observations are already grayscale: only a resize is needed.
    """

    def process_observation(self, observation):
        """Resize a grayscale frame to 84x84 (uint8 to save memory)."""
        return cv2.resize(observation, INPUT_SHAPE[1:],
                          interpolation=cv2.INTER_AREA)

    def process_state_batch(self, batch):
        """Scale a batch of frames to [0, 1] as float32."""
        return batch.astype('float32') / 255.0

    def process_reward(self, reward):
        """Clip the reward to [-1, 1]."""
        return max(-1.0, min(1.0, reward))


class CompatibilityWrapper(gym.Wrapper):
    """
    Makes Gymnasium compatible with keras-rl and helps learning:
    - step returns (obs, reward, done, info)
    - reset returns only the observation
    - a lost life ends the episode (faster credit assignment)
    - FIRE is pressed automatically to launch the ball
    """

    def __init__(self, env):
        """Initializes the wrapper and the life counter."""
        super().__init__(env)
        self.lives = 0
        self.game_over = True

    def step(self, action):
        """Steps the env; done is True on game end or on a lost life."""
        observation, reward, terminated, truncated, info = \
            self.env.step(action)
        self.game_over = terminated or truncated
        done = self.game_over or info['lives'] < self.lives
        self.lives = info['lives']
        return observation, reward, done, info

    def reset(self, **kwargs):
        """
        Fully resets only after a real game over, otherwise continues
        the current game. The ball is always launched with FIRE.
        """
        if self.game_over:
            self.env.reset(**kwargs)
        observation, _, _, _, info = self.env.step(1)
        self.lives = info['lives']
        self.game_over = False
        return observation

    def render(self, *args, **kwargs):
        """Renders without the legacy arguments used by keras-rl."""
        return self.env.render()


def build_model(input_shape, nb_actions):
    """
    Builds the convolutional Q-network (DeepMind architecture).

    Args:
        input_shape (tuple): shape of a stacked observation (4, 84, 84).
        nb_actions (int): number of possible actions.

    Returns:
        keras.models.Sequential: the Q-network.
    """
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=input_shape))
    model.add(Conv2D(32, (8, 8), strides=4, activation='relu'))
    model.add(Conv2D(64, (4, 4), strides=2, activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=1, activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(nb_actions, activation='linear'))
    return model


def build_agent(model, nb_actions):
    """
    Constructs the DQN agent with memory and policy settings.

    Args:
        model (keras.models.Sequential): the Q-network.
        nb_actions (int): number of possible actions.

    Returns:
        rl.agents.DQNAgent: compiled agent ready for training.
    """
    memory = SequentialMemory(limit=300000, window_length=4)
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr='eps',
        value_max=1.0,
        value_min=0.05,
        value_test=0.02,
        nb_steps=400000
    )
    dqn = DQNAgent(model=model,
                   nb_actions=nb_actions,
                   memory=memory,
                   nb_steps_warmup=20000,
                   target_model_update=10000,
                   processor=AtariProcessor(),
                   gamma=0.99,
                   policy=policy,
                   train_interval=4,
                   delta_clip=1.0,
                   enable_double_dqn=True)
    dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])
    return dqn


def main():
    """
    Creates the environment, builds the agent, trains it and
    saves the final policy network to policy.h5.
    """
    env = CompatibilityWrapper(
        gym.make('ALE/Breakout-v5', obs_type='grayscale'))
    nb_actions = env.action_space.n
    model = build_model(INPUT_SHAPE, nb_actions)
    dqn = build_agent(model, nb_actions)

    callbacks = [ModelIntervalCheckpoint('checkpoint_{step}.h5f',
                                         interval=100000)]
    dqn.fit(env, nb_steps=1500000, visualize=False, verbose=2,
            callbacks=callbacks)
    dqn.save_weights('policy.h5', overwrite=True)
    env.close()


if __name__ == '__main__':
    main()