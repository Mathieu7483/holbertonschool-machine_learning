#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def line():
    """
    Plots y = x^3 as a red line
    """
    x = np.arange(0, 11)
    y = x ** 3

    # On définit la figure AVANT le plot
    plt.figure(figsize=(6.4, 4.8))

    # On passe explicitement x et y, avec le formatage 'r-'
    plt.plot(x, y, 'r-')

    # Configuration des labels et limites
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('y = x^3')
    
    # On s'assure que les limites sont bien des entiers ou flottants exacts
    plt.xlim(0, 10)
    plt.ylim(0, 1000)

    plt.show()
