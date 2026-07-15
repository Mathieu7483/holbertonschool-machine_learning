<p align="center"\>
<img src="https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/unsupervised_learning/a-realistic--cinematic-photograph-of-a-machine-lea.png"\>
</p>


# Unsupervised Learning — Dimensionality Reduction

## 📉 Description

Ce projet est consacré à l'étude et à l'implémentation d'algorithmes de réduction de dimensionnalité. Dans les scénarios réels de Machine Learning, les données souffrent souvent du fléau de la dimensionnalité (*curse of dimensionality*). Réduire le nombre de caractéristiques tout en préservant l'information essentielle est crucial pour accélérer l'entraînement des modèles et permettre la visualisation des données complexes.

Ce dépôt regroupe l'implémentation de deux piliers du domaine :

1. **PCA (Principal Component Analysis)** : Une technique linéaire de projection orthogonale maximisant la variance.
2. **t-SNE (t-distributed Stochastic Neighbor Embedding)** : Une technique non linéaire d'alignement de distributions de probabilités, idéale pour projeter des collectifs de données complexes (manifolds) sur des espaces à 2 ou 3 dimensions.

## 🎓 Objectifs d'apprentissage

* **Décomposition de matrices** : Maîtriser la différence pratique et théorique entre l'écomposition spectrale (*Eigendecomposition*) et la décomposition en valeurs singulières (*SVD*).
* **Analyse en Composantes Principales (PCA)** : Comprendre comment projeter des données sur les vecteurs propres maximisant la variance globale.
* **t-SNE & Géométrie des données** :
* Appréhender le concept de *manifold* (variété topologique).
* Calculer les affinités conditionnelles basées sur la distribution normale ($P$) et de Student ($Q$).
* Minimiser la divergence de Kullback-Leibler (KL) par descente de gradient pour optimiser le positionnement des points en basse dimension.



## 🛠️ Spécifications techniques

* **OS** : Ubuntu 20.04 LTS | **Python** : 3.9
* **Framework** : NumPy (1.25.2).
* **Style** : Respect strict des standards de la norme `pycodestyle` (version 2.11.1).
* **Contrainte d'optimisation** : Minimiser le nombre d'opérations pour limiter la dérive des calculs en virgule flottante (*floating point errors*).

---

## 📂 Pipeline du Projet

Le projet se divise en deux grandes phases (9 tâches au total, de 0 à 8) :

### Part 1 : Analyse en Composantes Principales (Linear)

* **`0-pca.py`** : Réduction de dimension par PCA en maintenant une fraction minimale de la variance cumulée originale (`var`), à l'aide de la décomposition SVD.
* **`1-pca.py`** : Version alternative projetant les données directement sur un nombre fixe de dimensions visées.

### Part 2 : Implémentation complète de t-SNE (Non-linear)

* **`2-multivariate_gaussian.py`** : Initialisation des variables de l'algorithme t-SNE.
* **`3-entropy.py` / `4-probability.py**` : Calcul des affinités $P$ dans l'espace d'origine et ajustement de la perplexité (entropie de Shannon).
* **`5-decision_boundary.py`** : Calcul des affinités $Q$ de la distribution t-Student dans l'espace de basse dimension.
* **`6-gradients.py` / `7-cost.py**` : Calcul du coût de la divergence KL et de son gradient pour ajuster les coordonnées.
* **`8-tsne.py`** : Assemblage final de la boucle d'optimisation t-SNE.

---

## 🔬 Focus Mathématique : SVD vs EIG pour la PCA (Tâche 0)

Pour effectuer la PCA, l'usage de la décomposition en valeurs singulières (SVD) sur la matrice de données centrées $X$ est généralement plus stable numériquement et plus rapide que le calcul de l'écomposition spectrale (EIG) de la matrice de covariance $X^T X$.

La décomposition SVD s'écrit :

$$X = U \Sigma V^T$$

Où :

* $U$ contient les vecteurs singuliers à gauche.
* $\Sigma$ contient les valeurs singulières (liées aux valeurs propres par $\lambda_i = \frac{s_i^2}{n-1}$).
* $V^T$ contient les vecteurs singuliers à droite, qui correspondent exactement aux axes principaux (les poids de notre matrice de projection $W$).

Pour conserver une fraction de variance $v_{\text{target}}$ (ex: 95%), on calcule la somme cumulée des variances expliquées :

$$\text{Variance Expliquée}_i = \frac{s_i^2}{\sum_{j=1}^d s_j^2}$$

On sélectionne les $r$ premiers vecteurs de $V$ tels que la somme cumulée de leur variance expliquée atteigne ou dépasse $v_{\text{target}}$.

---

## 🛠️ Validation de la Tâche 0 : PCA

Le script d'exemple projette un jeu de données généré aléatoirement et vérifie l'erreur de reconstruction globale :

```bash
chmod +x 0-main.py
./0-main.py

```

### Sortie attendue (extrait) :

```text
[[-16.71379391   3.25277063  -3.21956297]
 ...
 [  7.38044431  -1.58972122   0.60154138]]
1.7353180054998176e-29

```

*(Une erreur de reconstruction de l'ordre de $10^{-29}$ confirme la perfection mathématique de la décomposition).*

## ✍️ Auteur

  * **Mathieu** - *Programming student, specialization Machine Learning* - [👤 My Github profile](https://github.com/Mathieu7483)
