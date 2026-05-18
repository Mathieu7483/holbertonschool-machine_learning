# Convolutions and Pooling

## 📸 Description

Ce projet explore les fondements mathématiques et algorithmiques du traitement d'images et des réseaux de neurones convolutifs (CNN). L'objectif est d'implémenter à la main, uniquement avec **NumPy** et sans utiliser de fonctions haut niveau (comme `np.convolve`), les mécanismes de produit de convolution et de réduction spatiale (**Pooling**). Ce travail met en lumière l'impact des hyperparamètres clés : taille du noyau (kernel), pas de déplacement (stride), et gestion des bordures (padding).

## 🎓 Objectifs d'apprentissage

* **Mécanisme** : Comprendre le produit scalaire glissant entre une image et un noyau (filter).
* **Hyperparamètres** : Maîtriser l'impact du *Stride* et faire la distinction entre le padding `valid` (aucune bordure) et `same` (préservation des dimensions).
* **Dimensions** : Manipuler des volumes complexes impliquant plusieurs canaux (RGB) et plusieurs noyaux simultanément.
* **Réduction** : Implémenter le Max Pooling et l'Average Pooling pour compresser l'information spatiale tout en conservant les caractéristiques importantes.

## 🛠️ Spécifications techniques

* **OS** : Ubuntu 20.04 LTS | **Python** : 3.9
* **Bibliothèque** : NumPy (1.25.2) exclusivement.
* **Style** : Conformité stricte avec `pycodestyle` (2.11.1).
* **Contrainte algorithmique** : L'utilisation de boucles est strictement limitée (maximum deux boucles `for` selon les tâches). L'usage de `np.convolve` est interdit.

---

## 📂 Liste des Tâches (Convolutions)

| # | Tâche | Fichier | Description |
| --- | --- | --- | --- |
| **0** | **Valid Convolution** | `0-convolve_grayscale_valid.py` | Convolution sur images en niveaux de gris sans padding. |
| **1** | **Same Convolution** | `1-convolve_grayscale_same.py` | Convolution avec padding calculé pour maintenir les dimensions d'origine. |
| **2** | **Convolution with Padding** | `2-convolve_grayscale_padding.py` | Convolution sur images avec un padding personnalisé explicite. |
| **3** | **Strided Convolution** | `3-convolve_grayscale.py` | Convolution prenant en compte un pas de déplacement (*stride*) variable. |
| **4** | **Convolution with Channels** | `4-convolve_channels.py` | Convolution sur des images à plusieurs canaux (ex: RGB). |
| **5** | **Multiple Kernels** | `5-convolve.py` | Convolution appliquant plusieurs filtres différents sur une image à canaux. |

## 🚀 Tâches Avancées (Pooling)

| # | Tâche | Fichier | Description |
| --- | --- | --- | --- |
| **6** | **Pooling** | `6-pool.py` | Implémentation complète du Max Pooling et de l'Average Pooling sur des images à canaux. |

---

## 🔬 Focus : Calcul des dimensions de sortie

Pour une image d'entrée de taille $(H, W)$, un noyau de taille $(K_h, K_w)$, un padding $P$ et un stride $S$, les dimensions de la matrice de sortie se calculent ainsi :

$$H_{out} = \lfloor\frac{H + 2P - K_h}{S}\rfloor + 1$$

$$W_{out} = \lfloor\frac{W + 2P - K_w}{S}\rfloor + 1$$

* En mode **Valid** : $P = 0$.
* En mode **Same** : $P$ est calculé dynamiquement pour que $H_{out} = H$ et $W_{out} = W$ (lorsque $S = 1$).

## ✍️ Auteur

* **Mathieu** - *Étudiant en programmation (42 ans)* - [GitHub Profile](https://www.google.com/search?q=https://github.com/Mathieu7483)
