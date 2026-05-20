# Convolutional Neural Networks (CNNs)

## 🧠 Description

Ce projet est dédié à la compréhension profonde et à l'implémentation bas niveau des réseaux de neurones convolutifs. Au-delà de l'utilisation d'API de haut niveau, l'objectif ici est de coder en pur **NumPy** l'ensemble du flux mathématique d'un CNN, y compris la rétropropagation (*Backpropagation*) à travers les couches de convolution et de pooling. Le projet se conclut par l'utilisation de **Keras** pour implémenter l'architecture classique LeNet-5, marquant la transition entre la théorie pure et l'ingénierie pratique en Deep Learning.

## 🎓 Objectifs d'apprentissage

* **Forward Propagation** : Calculer les sorties de volumes convolutifs et de pooling (Max/Average) à partir de tenseurs à 4 dimensions $(m, h, w, c)$.
* **Backpropagation (Bas niveau)** : Calculer analytiquement les gradients des pertes par rapport aux poids ($\partial W$), aux biais ($\partial b$) et aux activations précédentes ($\partial A_{prev}$).
* **Keras & TensorFlow** : Construire, compiler et entraîner une architecture CNN standardisée en utilisant l'API Keras.
* **Reproductibilité** : Maîtriser la fixation des graines aléatoires (*seeds*) pour garantir des entraînements déterministes.

## 🛠️ Spécifications techniques

* **OS** : Ubuntu 20.04 LTS | **Python** : 3.9
* **Bibliothèques** : NumPy (1.25.2) et TensorFlow (2.15).
* **Style** : Conformité absolue avec la norme `pycodestyle` (2.11.1).
* **Structure** : Tous les fichiers de scripts doivent être exécutables et documentés (modules, classes et fonctions).

---

## 📂 Liste des Tâches (Numpy - Scratch Implementation)

| # | Tâche | Fichier | Description |
| --- | --- | --- | --- |
| **0** | **Convolutional Forward Prop** | `0-conv_forward.py` | Propagation avant complète d'une couche convolutive avec biais et fonction d'activation. |
| **1** | **Pooling Forward Prop** | `1-pool_forward.py` | Propagation avant d'une couche de pooling (Max ou Average). |
| **2** | **Convolutional Back Prop** | `2-conv_backward.py` | Calcul des gradients et rétropropagation à travers une couche convolutive. |
| **3** | **Pooling Back Prop** | `3-pool_backward.py` | Rétropropagation des gradients à travers une couche de Max ou Average Pooling. |

## 🚀 Tâches Avancées (Keras & Architecture)

| # | Tâche | Fichier | Description |
| --- | --- | --- | --- |
| **4** | **LeNet-5 (Keras)** | `4-lenet5.py` | Construction et compilation du modèle LeNet-5 modifié via Keras (Dense, Conv2D, MaxPooling2D, Flatten). |
| **5** | **Summarize Like a Pro** | `5-lenet5.py` | Adaptation de l'architecture LeNet-5 pour retourner le modèle compilé et afficher son résumé structurel. |

---

## 🔬 Focus Mathématique : Les Gradients Convolutifs

Pendant la rétropropagation de la couche convolutive (Tâche 2), pour chaque exemple $i$ et chaque élément de la matrice de sortie, le calcul des gradients accumule les produits des activations de la propagation avant et des gradients de la couche suivante ($\partial Z$) :

$$\partial W += \sum_{i=0}^{m} X_{slice}^{(i)} \times \partial Z_{i, h, w, c}$$

$$\partial b += \sum_{i=0}^{m} \partial Z_{i, h, w, c}$$

Le calcul de $\partial A_{prev}$ nécessite de redistribuer l'erreur sur chaque pixel de la fenêtre d'entrée en fonction des poids du noyau associé.

## ✍️ Auteur

* **Mathieu** - *Étudiant en programmation (42 ans)* - [GitHub Profile](https://www.google.com/search?q=https://github.com/Mathieu7483)
