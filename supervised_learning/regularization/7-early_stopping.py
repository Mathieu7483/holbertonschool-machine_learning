#!/usr/bin/env python3
"""Write the function that determines if you should stop gradient
descent early"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    Determines if you should stop gradient descent early
    Parameters:
        cost (float): The current cost of the model.
        opt_cost (float): The lowest cost of the model up to now.
        threshold (float): The threshold used to determine if the cost is
            optimal.
        patience (int): The patience used to determine if you should stop
            gradient descent early.
        count (int): The number of iterations that the cost has not been
            optimal.
    Returns:
        bool: True if you should stop gradient descent early, False otherwise.
    """
    if cost < opt_cost - threshold:
        count = 0
    else:
        count += 1

    if count >= patience:
        return True, count
    return False, count
