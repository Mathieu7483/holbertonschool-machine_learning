#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

def line():
    # 1. Préparation des données
    x = np.arange(0, 11)
    y = x ** 3

    # 2. Initialisation de la figure (DOIT être en premier)
    plt.figure(figsize=(6.4, 4.8))

    # 3. Tracé avec X et Y explicites + Couleur rouge
    plt.plot(x, y, 'r-')

    # 4. Configuration des axes (L'ordre xlabel/ylabel/title importe peu ici)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('y = x^3')

    # 5. Définition des limites (Utilise x[0] et x[-1] pour être exact)
    plt.xlim(0, 10)
    plt.ylim(0, 1000)

    # 6. Affichage
    plt.show()
