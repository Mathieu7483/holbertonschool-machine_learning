#!/usr/bin/env python3
"""Module for plotting a stacked bar chart of fruit quantities."""
import numpy as np
import matplotlib.pyplot as plt


def bars():
    """Plots a stacked bar chart of fruit quantities for three people."""
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    names = ['Farrah', 'Fred', 'Felicia']
    fruits = ['apples', 'bananas', 'oranges', 'peaches' ]
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

    indices = np.arange(len(names))
    width = 0.5
    bottom = np.zeros(len(names))

    for i in range(len(fruits)):
        plt.bar(indices, fruit[i], bottom=bottom, color=colors[i],
                label=fruits[i])
        bottom += fruit[i]
    plt.ylabel('Quantity of Fruit')
    plt.title('Number of Fruit per Person')
    plt.xticks(indices, names)
    plt.yticks(range(0, 81, 10))
    plt.legend(loc='upper right')
    plt.show()
