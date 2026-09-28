<p align="center"\>
<img src="https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/reinforcement_learning/q_learning/Q%20learning.png"\>
</p>

# Q-Learning — Gymnasium FrozenLake-v1

## Description

Ce projet aborde les fondements du **Reinforcement Learning (RL)** tabulaire en implémentant l'algorithme du **Q-Learning** à partir de zéro (*from scratch*) sous Python avec **Gymnasium**.

L'objectif est d'entraîner un agent autonome à naviguer dans l'environnement **FrozenLake-v1** (carte sous forme de grille composée de glace sûre `S`/`F`, de trous mortels `H` et d'un objectif `G`). Le projet couvre l'ensemble du processus :

* Modélisation sous forme de **Processus de Décision Markovien (MDP)**.
* Initialisation de la **Q-Table** pour représenter la fonction de valeur action-état $Q(s, a)$.
* Arbitrage entre exploration et exploitation via la stratégie **$\epsilon$-greedy**.
* Mise à jour itérative des Q-valeurs selon l'équation de **Bellman**.
* Évaluation de l'agent en mode exploitation pure (*Play*).

---

## Technical Requirements

* **OS:** Ubuntu 20.04 LTS
* **Language:** Python 3.9
* **Main Libraries:**
* NumPy 1.25.2
* Gymnasium 0.29.1 (`gymnasium`)
* Pillow 10.3.0
* h5py 3.11.0


* **Style Guide:** Conformité stricte aux normes `pycodestyle` (v2.11.1)
* **Executable:** Tous les scripts exécutables commencent par `#!/usr/bin/env python3`
* **Documentation:** Tous les modules, classes et fonctions sont intégralement documentés.

---

## Environment Setup & Installation

### Installation des dépendances

```bash
pip install --user gymnasium==0.29.1 Pillow==10.3.0 h5py==3.11.0

```

---

## Key Concepts & Mathematics

* **Markov Decision Process (MDP) :** Un environnement défini par un ensemble d'états $S$, d'actions $A$, de probabilités de transition $P(s' \vert{} s, a)$ et de récompenses $R(s, a)$.
* **Q-Table Initialization :** Une matrice de dimensions $(\vert{}S\vert{}, \vert{}A\vert{})$ initialisée à zéro, stockant la qualité estimée de chaque action dans chaque état.
* **Exploration vs. Exploitation ($\epsilon$-greedy) :**
Avec une probabilité $\epsilon$, l'agent choisit une action aléatoire (exploration). Avec une probabilité $1 - \epsilon$, il choisit l'action maximisant la Q-valeur actuelle :

$$\text{Action} = \arg\max_{a} Q(s, a)$$

* **Bellman Equation & Q-Update :**
L'algorithme de Q-Learning utilise l'Équation de Bellman de programmation dynamique pour ajuster la Q-valeur après chaque transition $(s, a, r, s')$ :

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

* $\alpha$ (*learning rate*) : Taux d'apprentissage.
* $\gamma$ (*discount factor*) : Facteur de remise pour privilégier les gains à court ou long terme.

---

## File Structure & Tasks Overview

| File | Description | Key Components / Concepts |
| --- | --- | --- |
| `0-load_env.py` | Chargement et configuration de l'environnement Gymnasium `FrozenLake-v1`. | Custom map (`desc`), `map_name`, Deterministic vs Slippery mode |
| `1-q_init.py` | Initialisation de la Q-table aux dimensions de l'environnement. | State space `env.observation_space.n`, Action space `env.action_space.n` |
| `2-epsilon_greedy.py` | Implémentation de la politique de sélection d'action $\epsilon$-greedy. | Exploration (`np.random.uniform`), Exploitation (`np.argmax`) |
| `3-q_learning.py` | Algorithme complet d'entraînement du Q-Learning sur plusieurs épisodes. | Q-table update loop, Epsilon decay, Reward tracking |
| `4-play.py` | Test et évaluation de l'agent entraîné (mode exploitation pure). | Greedy action selection, Render / Step simulation |

---

## Usage Example (Task 0 - Load Environment)

```bash
chmod +x 0-main.py 0-load_env.py
./0-main.py

```

### Script Example (`0-main.py`)

```python
#!/usr/bin/env python3
load_frozen_lake = __import__('0-load_env').load_frozen_lake

# Chargement d'une carte 4x4 non glissante
env = load_frozen_lake(map_name='4x4', is_slippery=False)
print("Description de la grille :")
print(env.unwrapped.desc)

# Chargement d'une carte personnalisée
desc = [['S', 'F', 'F'], 
        ['F', 'H', 'H'], 
        ['F', 'F', 'G']]
env_custom = load_frozen_lake(desc=desc)
print("\nDescription carte sur-mesure :")
print(env_custom.unwrapped.desc)

```

---

## ✍️ Author

  * **Mathieu** - *Programming student, specialization Machine Learning* - [👤 My Github profile](https://github.com/Mathieu7483)