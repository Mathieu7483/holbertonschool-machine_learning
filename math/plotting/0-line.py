#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def line():
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('y = x^3')
    plt.xlim(0, 10)
    plt.ylim(0, 1000)
    plt.show()
