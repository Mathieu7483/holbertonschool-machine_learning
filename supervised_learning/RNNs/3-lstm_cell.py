#!usr/bin/env python3
"""Create the class LSTMCell that represents an LSTM unit"""
import numpy as np


class LSTMCell:
    """Represent an LSTM unit"""

    def __init__(self, i, h, o):
        """Initialize the LSTM cell

        Args:
            i (int): The dimensionality of the data
            h (int): The dimensionality of the hidden state
            o (int): The dimensionality of the outputs
        """

        self.Wf = np.random.normal(size=(i + h, h))
        self.bf = np.zeros((1, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.bu = np.zeros((1, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.bc = np.zeros((1, h))
        self.Wo = np.random.normal(size=(i + h, o))
        self.bo = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """Perform forward propagation for one time step

        Args:
            h_prev (numpy.ndarray): The previous hidden state
            c_prev (numpy.ndarray): The previous cell state
            x_t (numpy.ndarray): The data input for the cell

        Returns:
            h_next (numpy.ndarray): The next hidden state
            c_next (numpy.ndarray): The next cell state
        """

        # Concatenate h_prev and x_t
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Forget gate
        f_t = self.sigmoid(np.dot(concat, self.Wf) + self.bf)

        # Update gate
        u_t = self.sigmoid(np.dot(concat, self.Wu) + self.bu)

        # Candidate cell state
        c_hat_t = np.tanh(np.dot(concat, self.Wc) + self.bc)

        # Next cell state
        c_next = f_t * c_prev + u_t * c_hat_t

        # Output gate
        o_t = self.sigmoid(np.dot(concat, self.Wo) + self.bo)

        # Next hidden state
        h_next = o_t * np.tanh(c_next)

        return h_next, c_next
