<p align="center"\>
<img src="https://github.com/Mathieu7483/Aiko78-Photgraphy/blob/main/img/Machine%20Learning.png"\>
</p>

# Optimization: Speed up your Neural Networks

## 🚀 Description
Ce projet se concentre sur les techniques avancées pour accélérer l'entraînement des réseaux de neurones et améliorer leur performance. On y explore la mise à l'échelle des données (**Feature Scaling**), les algorithmes de descente de gradient optimisés (**Momentum**, **RMSProp**, **Adam**), ainsi que la **Batch Normalization** et le **Learning Rate Decay**. L'enjeu est de naviguer efficacement dans l'espace des paramètres pour éviter les points de selle et atteindre les minima locaux/globaux plus rapidement.

## 🎓 Objectifs d'apprentissage
* **Normalisation** : Savoir pourquoi et comment normaliser les entrées pour faciliter la descente de gradient.
* **Mini-Batch** : Implémenter le Mini-Batch Gradient Descent pour un meilleur compromis vitesse/stabilité.
* **Moyennes Mobiles** : Comprendre les moyennes pondérées exponentielles, socle des algorithmes modernes.
* **Optimiseurs** : Maîtriser le fonctionnement interne de Momentum, RMSProp et Adam.
* **Schedules** : Apprendre à réduire le taux d'apprentissage au fil du temps (Decay).
* **Batch Normalization** : Stabiliser l'apprentissage en normalisant les activations entre les couches.

## 🛠️ Spécifications techniques
* **OS** : Ubuntu 20.04 LTS | **Python** : 3.9
* **Bibliothèques** : NumPy (1.25.2) et TensorFlow (2.15).
* **Style** : Conformité totale avec `pycodestyle` (2.11.1).
* **Documentation** : Chaque module, classe et fonction doit être documenté.

---

## 📂 Liste des Tâches (Foundations & Algorithms)

| # | Tâche | Fichier | Description |
| :--- | :--- | :--- | :--- |
| **0** | **Norm Constants** | `0-norm_constants.py` | Calcul de la moyenne et de l'écart-type. |
| **1** | **Normalize** | `1-normalize.py` | Standardisation d'une matrice de données. |
| **2** | **Shuffle Data** | `2-shuffle_data.py` | Mélange aléatoire des données pour le Mini-batch. |
| **3** | **Mini-Batch** | `3-mini_batch.py` | Découpage des données en petits lots d'entraînement. |
| **4** | **Moving Average** | `4-moving_average.py` | Calcul d'une moyenne mobile exponentielle. |
| **5** | **Momentum** | `5-momentum.py` | Implémentation du Gradient Descent avec Momentum. |
| **6** | **Momentum TF** | `6-momentum.py` | Version TensorFlow de l'optimiseur Momentum. |
| **7** | **RMSProp** | `7-rmsprop.py` | Implémentation manuelle de RMSProp. |
| **8** | **RMSProp TF** | `8-rmsprop.py` | Version TensorFlow de l'optimiseur RMSProp. |

## 🚀 Tâches Avancées (Adam & Batch Norm)

| # | Tâche | Fichier | Description |
| :--- | :--- | :--- | :--- |
| **9** | **Adam** | `9-adam.py` | Implémentation manuelle complète de l'algorithme Adam. |
| **10** | **Adam TF** | `10-adam.py` | Utilisation de l'optimiseur Adam via TensorFlow. |
| **11** | **LR Decay** | `11-learning_rate_decay.py` | Calcul manuel de la réduction du taux d'apprentissage. |
| **12** | **LR Decay TF** | `12-learning_rate_decay.py` | Implémentation du decay avec TensorFlow. |
| **13** | **Batch Norm** | `13-batch_norm.py` | Normalisation de couche manuelle. |
| **14** | **Batch Norm TF** | `14-batch_norm.py` | Implémentation de la Batch Normalization avec TF. |
| **15** | **Blog Post** | `N/A` | Synthèse technique sur les concepts d'optimisation. |

---

## 🔬 Focus : L'algorithme Adam
**Adam** (Adaptive Moment Estimation) est le roi des optimiseurs. Il combine les avantages de **Momentum** (en gardant une trace de la direction passée) et de **RMSProp** (en ajustant le taux d'apprentissage pour chaque paramètre). C'est l'outil indispensable pour tout projet de Deep Learning sérieux.

## ✍️ Author

  * **Mathieu** - *Programming student, specialization Machine Learning* - [👤 My Github profile](https://github.com/Mathieu7483)