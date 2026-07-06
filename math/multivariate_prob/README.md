<p align="center"\>
<img src="https://github.com/Mathieu7483/Aiko78-Photgraphy/blob/main/img/Machine%20Learning.png"\>
</p>

# Mathematics — Multivariate Probability

## 📊 Description

Ce projet est dédié à l'étude mathématique et à l'implémentation algorithmique des distributions de probabilités multivariées, avec un focus particulier sur la **distribution Gaussienne multivariée**. L'objectif est de comprendre comment mesurer et modéliser les relations linéaires entre plusieurs dimensions (ou caractéristiques) d'un jeu de données. Le projet implique de coder manuellement les calculs de vecteurs de moyennes, de matrices de covariance et de coefficients de corrélation, avant de construire une classe complète capable de calculer la fonction de densité de probabilité (PDF) dans un espace multidimensionnel.

## 🎓 Objectifs d'apprentissage

* **Distributions Conjointes & Multivariées** : Étendre les concepts de probabilités univariées à des espaces à $d$ dimensions.
* **Matrice de Covariance** : Comprendre comment elle capture la variance de chaque caractéristique sur sa diagonale et la covariance inter-variable sur ses autres éléments.
* **Corrélation** : Standardiser la covariance pour obtenir des coefficients de corrélation limités entre $-1$ et $1$.
* **Loi Normale Multivariée** : Manipuler et implémenter sa formule de densité en utilisant le déterminant et l'inverse de la matrice de covariance.

## 🛠️ Spécifications techniques

* **OS** : Ubuntu 20.04 LTS | **Python** : 3.9
* **Framework** : NumPy (1.25.2).
* **Style** : Respect strict et systématique des règles de la norme `pycodestyle` (2.11.1).
* **Restrictions** : Interdiction absolue d'utiliser la fonction intégrée `numpy.cov`. Tout calcul de covariance doit être vectorisé manuellement via NumPy.

---

## 📂 Liste des Tâches et Pipeline

| # | Tâche | Fichier | Description |
| --- | --- | --- | --- |
| **0** | **Mean and Covariance** | `0-mean_cov.py` | Calcule le vecteur des moyennes et la matrice de covariance d'un jeu de données de taille $n \times d$. |
| **1** | **Correlation** | `1-correlation.py` | Calcule la matrice des coefficients de corrélation à partir d'une matrice de covariance. |
| **2** | **Initialize** | `2-multivariate_gaussian.py` | Constructeur de la classe `MultivariateGaussian` qui estime et stocke la moyenne et la covariance d'un échantillon. |
| **3** | **PDF** | `3-multivariate_gaussian.py` | Calcule la valeur de la PDF (densité de probabilité) pour un point donné de l'espace. |

---

## 🔬 Focus Mathématique : Calcul Vectorisé de la Covariance

Pour un jeu de données représenté par une matrice $X$ de forme $(n, d)$, où $n$ est le nombre d'échantillons et $d$ le nombre de dimensions, la formule empirique de la matrice de covariance $\Sigma$ (de taille $d \times d$) est définie par :

$$\Sigma = \frac{1}{n - 1} (X - \mu)^T (X - \mu)$$

Où $\mu$ représente le vecteur ligne des moyennes de chaque dimension (de taille $1 \times d$).

> ⚠️ **Attention au biais** : Le dénominateur utilise $n - 1$ (correction de Bessel) et non $n$ pour obtenir un estimateur non biaisé de la covariance de la population à partir d'un échantillon.

---

## 🛠️ Validation de la Tâche 0

Le script de validation vérifie que les résultats coïncident parfaitement avec une distribution normale multivariée simulée :

```bash
chmod +x 0-main.py
./0-main.py

```

### Sortie attendue :

```text
[[12.04341828 29.92870885 10.00515808]]
[[ 36.2007391  -29.79405239  15.37992641]
 [-29.79405239  97.77730626 -20.67970134]
 [ 15.37992641 -20.67970134  24.93956823]]

```

## ✍️ Auteur

* **Mathieu** - *Étudiant en apprentissage Machine Learning (42 ans)* - [GitHub Profile](https://www.google.com/search?q=https://github.com/Mathieu7483)

