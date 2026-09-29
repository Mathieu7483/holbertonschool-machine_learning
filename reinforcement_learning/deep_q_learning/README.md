<p align="center"\>
<img src="https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/reinforcement_learning/deep_q_learning/deep%20Q%20learning.png"\>
</p>

# Deep Q-Learning — Atari Breakout Agent

## Description

Ce projet aborde le **Deep Reinforcement Learning (DRL)** à travers la conception, l'entraînement et l'évaluation d'un agent autonome capable de jouer au jeu classique **Atari 2600 Breakout**.

En combinant le Q-Learning tabulaire avec la puissance d'approximation des **réseaux de neurones convolutifs (CNN)**, l'algorithme **Deep Q-Network (DQN)** résout les limites liées à la dimensionnalité élevée des espaces d'états visuels (pixels d'écran). Le projet met en œuvre :

* L'intégration d'un environnement **Gymnasium Atari** (`BreakoutNoFrameskip-v4`) avec prétraitement des images (*Wrappers* de redimensionnement, passage en niveaux de gris et empilement de frames).
* L'utilisation de la bibliothèque **`keras-rl2`** pour gérer les composants clés de l'architecture DQN : mémoire de rejeu (*Replay Memory*), politique $\epsilon$-greedy et réseau cible (*Target Network*).
* La sauvegarde des poids du réseau de neurones (`policy.h5`) et l'évaluation de l'agent en mode exploitation pure (*Greedy Policy*).

---

## Technical Requirements

* **OS:** Ubuntu 20.04 LTS
* **Language:** Python 3.9
* **Main Libraries:**
* TensorFlow 2.15.0 & Keras 2.15.0
* Keras-RL2 1.0.4 (`keras-rl2`)
* Gymnasium 0.29.1 (`gymnasium[atari]`)
* NumPy 1.25.2
* Pillow 10.3.0 & h5py 3.11.0
* AutoROM (`autorom[accept-rom-license]`)


* **Style Guide:** Conformité stricte aux normes `pycodestyle` (v2.11.1)
* **Executable:** Tous les scripts exécutables commencent par `#!/usr/bin/env python3`
* **Documentation:** Modules, classes et fonctions entièrement documentés.

---

## Environment Setup & Installation

### 1. Installation des dépendances Python

```bash
pip install --user tensorflow==2.15.0 keras==2.15.0 keras-rl2==1.0.4
pip install --user "gymnasium[atari]==0.29.1" numpy==1.25.2 Pillow==10.3.0 h5py==3.11.0

```

### 2. Téléchargement et installation des ROMs Atari

```bash
pip install autorom[accept-rom-license]
AutoROM --accept-license

```

---

## Key Concepts & Architecture

* **Deep Q-Network (DQN) :** Réseau de neurones convolutif utilisé pour approximer la fonction de valeur d'action $Q(s, a; \theta)$, où $s$ représente les états visuels complexes et $a$ l'action à effectuer.
* **Experience Replay (`SequentialMemory`) :** Stocke les transitions observées $(s, a, r, s', \text{done})$ dans un buffer circulaire. L'échantillonnage aléatoire de batchs lors de l'entraînement rompt la corrélation temporelle entre les données et stabilise l'apprentissage.
* **Target Network vs Policy Network :** Découplage de la sélection des actions de la mise à jour des valeurs Q. Un réseau cible (*Target Network*) gelé pendant $N$ étapes fournit des cibles stables pour l'équation de Bellman, évitant les oscillations et la divergence.
* **Gymnasium Environment Wrappers :** Adaptateurs personnalisés pour adapter les sorties de Gymnasium (`reset`, `step`, `render`) à l'interface attendue par `keras-rl2` (gestion des tuples d'état/info, normalisation des pixels et stacking de 4 images consécutives pour capturer la dynamique et la vitesse de la bille).

---

## File Structure & Tasks Overview

| File | Description | Key Components / Concepts |
| --- | --- | --- |
| `train.py` | Script d'entraînement complet de l'agent DQN sur Atari Breakout. | CNN Architecture, `SequentialMemory`, `EpsGreedyQPolicy`, `DQNAgent.fit()`, Output: `policy.h5` |
| `play.py` | Script de démonstration visuelle et d'évaluation de l'agent entraîné. | Load weights (`policy.h5`), `GreedyQPolicy`, Gymnasium render loop |

---

## Usage & Execution

### 1. Entraînement de l'agent

Lancer le script d'entraînement pour générer le fichier de poids du réseau de neurones (`policy.h5`) :

```bash
chmod +x train.py play.py
./train.py

```

### 2. Démonstration de l'agent

Exécuter le script de démonstration pour visualiser une partie jouée par l'agent entraîné :

```bash
./play.py

```

---

## ✍️ Author

  * **Mathieu** - *Programming student, specialization Machine Learning* - [👤 My Github profile](https://github.com/Mathieu7483)