<p align="center"\>
<img src="https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/reinforcement_learning/temporal_difference/temporal%20difference%20learning.png"\>
</p>

# Temporal Difference — Reinforcement Learning

## Description

Ce projet aborde l'apprentissage par renforcement (**Reinforcement Learning**) sans modèle (*Model-Free RL*) en comparant deux paradigmes majeurs d'estimation des fonctions de valeur : les méthodes de **Monte Carlo (MC)** et l'apprentissage par **Différence Temporelle (Temporal Difference / TD)**.

L'objectif est d'implémenter et de comprendre les mécanismes fondamentaux sous-jacents aux algorithmes de prédiction et de contrôle :

* **Monte Carlo** : Évaluation basée sur le retour cumulé réel ($G_t$) obtenu à la fin d'un épisode complet.
* **TD Learning & Bootstrapping** : Mise à jour des estimations dès l'étape suivante à partir de la récompense immédiate et de la valeur estimée de l'état futur $V(S_{t+1})$.
* **TD($\lambda$) & Traces d'éligibilité (*Eligibility Traces*)** : Unification des méthodes TD et Monte Carlo via un paramètre de lissage $\lambda$.
* **SARSA / SARSA($\lambda$)** : Algorithme de contrôle *On-Policy* apprenant la fonction d'action-valeur $Q(s, a)$.

---

## Technical Requirements

* **OS:** Ubuntu 20.04 LTS
* **Language:** Python 3.9
* **Main Libraries:**
* NumPy 1.25.2
* Gymnasium 0.29.1 (`gymnasium`)


* **Style Guide:** Conformité stricte à la norme `pycodestyle` (v2.11.1)
* **Executable:** Tous les fichiers exécutables commencent par `#!/usr/bin/env python3`
* **Documentation:** Modules, classes et fonctions entièrement documentés.

---

## Key Concepts & Mathematics

### 1. Monte Carlo vs. Temporal Difference (TD)

* **Monte Carlo Target** : Nécessite un épisode complet jusqu'à l'état terminal $T$.
$$V(S_t) \leftarrow V(S_t) + \alpha \left[ G_t - V(S_t) \right]$$


* **TD(0) Target (Bootstrapping)** : Met à jour la valeur immédiatement après un pas de temps.
$$V(S_t) \leftarrow V(S_t) + \alpha \left[ R_{t+1} + \gamma V(S_{t+1}) - V(S_t) \right]$$



### 2. Traces d'éligibilité & TD($\lambda$)

La trace d'éligibilité $E_t(s)$ conserve une mémoire des états récents pour propager la mise à jour de la récompense sur l'ensemble de la trajectoire :

$$E_t(s) = \gamma \lambda E_{t-1}(s) + \mathbf{1}(S_t = s)$$

### 3. On-Policy vs. Off-Policy

* **On-Policy (ex: SARSA)** : Évalue et améliore la politique même qui est utilisée pour prendre des décisions sur le terrain.
* **Off-Policy (ex: Q-Learning)** : Évalue une politique optimale (greedy) tout en explorant l'environnement avec une autre (ex: $\epsilon$-greedy).

---

## File Structure & Tasks Overview

| File | Description | Key Components / Concepts |
| --- | --- | --- |
| `0-monte_carlo.py` | Implémentation de l'algorithme Monte Carlo pour l'estimation de la fonction de valeur $V(S)$. | Full Episode Trajectory, Return calculation ($G_t$), Value update |
| `1-td_lambtha.py` | Implémentation de l'algorithme TD($\lambda$) utilisant les traces d'éligibilité. | Eligibility Traces $E_t$, TD Error $\delta_t$, Bootstrapping |
| `2-sarsa_lambtha.py` | Implémentation de l'algorithme de contrôle On-Policy SARSA($\lambda$). | Action-Value $Q(s, a)$, $\epsilon$-greedy policy, Eligibility Traces |

---

## Installation & Usage

### 1. Installation des dépendances

```bash
pip install --user gymnasium==0.29.1 numpy==1.25.2

```

### 2. Exemple d'exécution (Task 0 - Monte Carlo)

```bash
chmod +x 0-main.py 0-monte_carlo.py
./0-main.py

```

### 3. Vérification de la conformité du code

```bash
pycodestyle *.py
python3 -c 'print(__import__("0-monte_carlo").monte_carlo.__doc__)'

```

---

## Author

* **Mathieu** — *Machine Learning Student @ Holberton School* — [GitHub Profile](https://github.com/Mathieu7483)