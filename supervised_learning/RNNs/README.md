<p align="center"\>
<img src="https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/supervised_learning/RNNs/RNN.png"\>
</p>
---

# Recurrent Neural Networks (RNNs)

## Description

Ce projet porte sur l'implémentation *from scratch* (à partir de zéro) des architectes de réseaux de neurones récurrents à l'aide de **NumPy** uniquement. L'objectif est d'assimiler les mécanismes fondamentaux sous-jacents aux architectures de traitement de séquences, la gestion des états cachés, et de comprendre les défis théoriques liés au *Backpropagation Through Time* (BPTT).

Ce dépôt couvre la conception algorithmique des modules suivants :

* **Simple RNN Cell & Full Forward Propagation**
* **Gated Recurrent Unit (GRU) Cell**
* **Long Short-Term Memory (LSTM) Cell**
* **Deep RNN Architecture**
* **Bidirectional RNN (BRNN) Cell & Forward Pass**

---

## Technical Requirements

* **OS:** Ubuntu 20.04 LTS
* **Language:** Python 3.9
* **Main Library:** NumPy 1.25.2 (Aucune autre bibliothèque tierce autorisée)
* **Style Guide:** Conformité stricte aux standards `pycodestyle` (v2.11.1)
* **Executable:** Tous les scripts exécutables commencent par `#!/usr/bin/env python3`
* **Documentation:** Modules, classes et fonctions entièrement documentés.

---

## File Structure & Tasks Overview

| Tâche | Fichier à rendre | Sujet |
| --- | --- | --- |
| **0. RNN Cell** | `0-rnn_cell.py` | Une étape de propagation avant d'une cellule RNN simple. |
| **1. RNN** | `1-rnn.py` | Propagation avant sur l'ensemble d'une séquence temporelle. |
| **2. GRU Cell** | `2-gru_cell.py` | Une étape de propagation avant d'une cellule GRU. |
| **3. LSTM Cell** | `3-lstm_cell.py` | Une étape de propagation avant d'une cellule LSTM. |
| **4. Deep RNN** | `4-deep_rnn.py` | Forward pass sur un RNN profond multi-couches. |
| **5. Bidirectional Cell Forward** | `5-bi_cell_forward.py` | Une étape d'une cellule bidirectionnelle. |
| **6. Bidirectional Forward** | `6-bi_forward.py` | Forward pass complet d'un RNN bidirectionnel. |
| **7. Gradient Vanishing / Exploding** | `7-properties.txt` | Questions de théorie sur la disparition/explosion du gradient. |
| **8. RNN Backprop** | `8-bi_rnn.py` / `8-rnn_backprop.py` | Backpropagation Through Time (BPTT). |

---

## Mathematical Concepts & Definitions

* **Exploding Gradient Problem:** Survient lorsque les gradients deviennent extrêmement grands lors de la rétropropagation à travers le temps (BPTT), rendant les poids instables.
* **Vanishing Gradient Problem:** Survient lorsque les gradients deviennent quasi nuls à travers les étapes temporelles, empêchant le réseau d'apprendre des dépendances à long terme.
* **LSTM & GRU Mitigation:** Ces architectures utilisent des mécanismes de **portes (gates)** et un flux d'information additif (le *cell state* $c_t$ pour LSTM) permettant de préserver le gradient sur de longues séquences.

---

## Installation & Usage

### 1. Clonage du dépôt

```bash
git clone https://github.com/Mathieu7483/holbertonschool-machine_learning.git
cd holbertonschool-machine_learning/supervised_learning/RNNs

```

### 2. Test d'une cellule RNN (Exemple avec la Task 0)

Assure-toi que le fichier principal de test `0-main.py` est présent et exécutable :

```bash
chmod +x 0-main.py 0-rnn_cell.py
./0-main.py

```

### 3. Verification de la conformité du code

```bash
pycodestyle *.py
python3 -c 'print(__import__("0-rnn_cell").RNNCell.__doc__)'

```

---

## Author

* **Mathieu** - *Machine Learning Student @ Holberton School* - [GitHub Profile](https://www.google.com/search?q=https://github.com/Mathieu7483)
