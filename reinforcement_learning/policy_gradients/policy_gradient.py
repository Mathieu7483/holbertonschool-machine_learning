#!/usr/bin/env python3
"""Write a function that computes the policy with a weight of a matrix."""
import numpy as np


def policy(matrix, weight):
    """Computes the policy with a weight of a matrix.

    Args:
        matrix (numpy.ndarray): A numpy.ndarray of shape (m, n)
        containing the input matrix.
        weight (numpy.ndarray): A numpy.ndarray of shape (n, k)
        containing the weights.

    Returns:
        numpy.ndarray: A numpy.ndarray of shape (m, k)
        containing the computed policy (probabilities).
    """
    logits = np.dot(matrix, weight)

    exp_scores = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)


def policy_gradient(state, weight):
    """  Computes the Monte-Carlo policy gradient.

    Args:
        state (np.ndarray): Current state of the environment (shape (n,)).
        weight (np.ndarray): Weight matrix (shape (n, m)).

    Returns:
        tuple: (action, gradient) where:
               - action: int, selected action.
               - gradient: np.ndarray, gradient of shape (n, m).
    """
    state = state.reshape(1, -1)
    probs = policy(state, weight)
    action = np.random.choice(probs.shape[1], p=probs.flatten())
    action_one_hot = np.zeros_like(probs)
    action_one_hot[0, action] = 1
    gradient = np.dot(state.T, (action_one_hot - probs))

    return action, gradient
