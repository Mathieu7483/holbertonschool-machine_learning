#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def line():
    # Définition explicite de X et Y
    x = np.arange(0, 11)
    y = x ** 3

    # Initialisation de la figure
    plt.figure(figsize=(6.4, 4.8))

    # Tracer AVEC x et y explicitement, en rouge ('r')
    # L'argument '-' assure que c'est une ligne continue
    plt.plot(x, y, 'r-')

    # Labels et Titre (Vérifie bien qu'il n'y a pas d'espace en trop)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('y = x^3')

    # Limites strictes
    plt.xlim(0, 10)
    plt.ylim(0, 1000)

    # Affichage
    plt.show()
