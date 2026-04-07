#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def gradient():

    np.random.seed(5)

    x = np.random.randn(2000) * 10
    y = np.random.randn(2000) * 10
    z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))
    plt.figure(figsize=(6.4, 4.8))

    plt.scatter(x, y, c=z, cmap='viridis')
    plt.colorbar()
    plt.xlim(-30, 30)
    plt.ylim(-30, 30)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gradient Plot')
    plt.show()
