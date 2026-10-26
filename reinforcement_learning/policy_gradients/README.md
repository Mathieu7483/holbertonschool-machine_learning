<p align="center"\>
<img src="https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/reinforcement_learning/temporal_difference/temporal%20difference%20learning.png"\>
</p>

# Reinforcement Learning — Policy Gradients (REINFORCE)

## Description

Ce projet aborde les méthodes de **Policy Gradient** en apprentissage par renforcement (*Reinforcement Learning*). Contrairement aux approches basées sur les fonctions de valeur (comme le Q-Learning ou SARSA) qui peinent dans les espaces d'actions continus ou stochastiques, les méthodes de gradient de politique optimisent directement la politique de l'agent $\pi_\theta(a\vert{}s)$ paramétrée par un réseau de neurones ou des poids linéaires.

L'objectif est d'implémenter l'algorithme de Monte-Carlo Policy Gradient, connu sous le nom de **REINFORCE** (Williams, 1992), pour entraîner un agent à résoudre des environnements dynamiques au sein de **Gymnasium**.

---

## Technical Requirements

* **OS:** Ubuntu 20.04 LTS
* **Language:** Python 3.9
* **Main Libraries:**
* NumPy 1.25.2
* Gymnasium 0.29.1 (`gymnasium`)


* **Style Guide:** Conformité stricte aux normes `pycodestyle` (v2.11.1)
* **Executable:** Tous les fichiers exécutables commencent par `#!/usr/bin/env python3`
* **Documentation:** Modules, classes et fonctions entièrement documentés.

---

## Key Concepts & Mathematics

* **Policy ($\pi$) :** Une fonction qui mappe les états de l'environnement vers une distribution de probabilité sur les actions possibles : $\pi_\theta(a\vert{}s) = P(A=a \vert{} S=s; \theta)$.
* **Softmax Policy & Log-probabilities :** Pour un modèle linéaire, les probabilités d'actions sont calculées en appliquant la fonction softmax aux scores d'action (produit scalaire entre l'état et les poids) :
$$\pi(a\vert{}s) = \frac{e^{z_a}}{\sum_b e^{z_b}} \quad \text{où} \quad z = s \cdot W$$


* **REINFORCE Algorithm (Monte-Carlo Policy Gradient) :** L'algorithme met à jour les poids en fonction du retour total $G_t$ de l'épisode complet, en renforçant les actions qui ont mené à de bonnes récompenses :
$$\theta \leftarrow \theta + \alpha \sum_{t} \nabla_\theta \log \pi_\theta(a_t \vert{} s_t) G_t$$



---

## File Structure & Tasks Overview

| File | Description | Key Components / Concepts |
| --- | --- | --- |
| `policy_gradient.py` | Contient les fonctions de calcul de politique, de gradient de Monte-Carlo, et la boucle d'entraînement. | Softmax policy, action sampling, log-derivative, Monte-Carlo returns $G_t$ |
| `0-main.py` et suivants | Scripts de test et d'animation des performances de l'agent entraîné. | Gymnasium rendering, trajectory visualization |

---

## Installation & Usage

### 1. Installation des dépendances

```bash
pip install --user gymnasium==0.29.1 numpy==1.25.2

```

### 2. Exemple d'exécution (Task 0 - Simple Policy)

```bash
chmod +x 0-main.py policy_gradient.py
./0-main.py

```

### 3. Vérification de la conformité du code

```bash
pycodestyle *.py
python3 -c 'print(__import__("policy_gradient").policy.__doc__)'

```

---

## ✍️ Author

  * **Mathieu** - *Programming student, specialization Machine Learning* - [👤 My Github profile](https://github.com/Mathieu7483)