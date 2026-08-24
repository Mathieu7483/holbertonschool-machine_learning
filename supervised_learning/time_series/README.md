<p align="center"\>
<img src="https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/supervised_learning/time_series/time%20series%20forecasting.jpg"\>
</p>

---

# Time Series Forecasting — BTC Price Prediction

## Description

Ce projet porte sur le prétraitement de données temporelles brutes et la mise en place de modèles d'apprentissage profond à base de **réseaux de neurones récurrents (RNNs)** avec **TensorFlow / Keras** pour prédire l'évolution du cours du Bitcoin (BTC).

L'objectif principal est de construire une pipeline de données complète (`tf.data.Dataset`), de gérer le fenêtrage glissant (*sliding window*) sur les 24 dernières heures ($24 \times 60 = 1440$ minutes ou points de données) et de prédire la valeur de clôture (*Close*) à l'heure suivante ($t + 60$ min).

---

## Technical Requirements

* **OS:** Ubuntu 20.04 LTS
* **Language:** Python 3.9
* **Main Libraries:**
* NumPy 1.25.2
* TensorFlow 2.15
* Pandas 2.2.2


* **Style Guide:** Conformité stricte aux normes `pycodestyle` (v2.11.1)
* **Executable:** Tous les scripts exécutables commencent par `#!/usr/bin/env python3`
* **Documentation:** Les modules, classes et fonctions doivent être entièrement documentés.

---

## Key Concepts & Theory

* **Time Series Forecasting :** Modélisation de séries chronologiques ordonnées dans le temps pour prédire des valeurs futures basées sur des observations passées.
* **Stationary Process :** Une série temporelle dont les propriétés statistiques (moyenne, variance, autocovariance) ne dépendent pas du temps. La stationnarité facilite l'apprentissage des modèles.
* **Sliding Window (Fenêtre glissante) :** Technique consistant à découper la séquence temporelle en blocs d'entrée de taille fixe (ex: 24 heures de données passées) associés à une cible future (ex: le prix 1h plus tard).
* **Data Preprocessing :** Traitement des valeurs manquantes (NaNs), rééchantillonnage de la fréquence des données, sélection de caractéristiques utiles et normalisation/mise à l'échelle (*MinMax* ou *StandardScaler*).
* **`tf.data.Dataset` Pipeline :** API haut niveau de TensorFlow permettant un chargement, un fenêtrage et un préchargement (*prefetching*) efficaces des données directement en mémoire GPU/CPU.

---

## File Structure & Tasks Overview

| File | Description | Key Features |
| --- | --- | --- |
| `preprocess_data.py` | Nettoyage, réalignement temporel, gestion des NaNs, normalisation et création des fenêtres pour les jeux de données Coinbase & Bitstamp. | Pandas, resample, cleaning, scaling, export `.npz` ou `.pkl` |
| `forecast_btc.py` | Définition de la pipeline `tf.data.Dataset`, création de l'architecture RNN (SimpleRNN, GRU ou LSTM), entraînement et validation du modèle Keras. | TensorFlow 2.15, `tf.data`, MSE Loss, Model Checkpointing |

---

## Dataset Format

Les données brutes (Coinbase et Bitstamp) contiennent les colonnes suivantes au format horodaté par fenêtre de 60 secondes :

1. `Timestamp` (Unix time)
2. `Open` (Prix d'ouverture en USD)
3. `High` (Prix le plus haut en USD)
4. `Low` (Prix le plus bas en USD)
5. `Close` (Prix de clôture en USD)
6. `Volume_(BTC)` (Volume d'échange en BTC)
7. `Volume_(Currency)` (Volume d'échange en USD)
8. `Weighted_Price` (Prix moyen pondéré par le volume)

---

## Installation & Execution

### 1. Préparation du répertoire

```bash
cd holbertonschool-machine_learning/supervised_learning/time_series

```

### 2. Prétraitement des données brutes

```bash
chmod +x preprocess_data.py forecast_btc.py
./preprocess_data.py

```

### 3. Entraînement et évaluation du modèle

```bash
./forecast_btc.py

```

### 4. Vérification de la conformité du code

```bash
pycodestyle *.py
python3 -c 'print(__import__("preprocess_data").__doc__)'
python3 -c 'print(__import__("forecast_btc").__doc__)'

```

---

## Author

* **Mathieu** — *Machine Learning Student @ Holberton School* — [GitHub Profile](https://github.com/Mathieu7483)