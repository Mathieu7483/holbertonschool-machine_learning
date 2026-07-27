<p align="center"\>
<img src="https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/unsupervised_learning/hyperparameter_tuning/hyperparameter%20tuning.png"\>
</p>

---

# Unsupervised Learning — Hyperparameter Tuning & Bayesian Optimization

## 📊 Description

Ce projet est dédié à l'optimisation avancée des hyperparamètres en Machine Learning. Lorsque l'évaluation d'un modèle (la fonction "boîte noire") est coûteuse en temps ou en ressources de calcul, les méthodes classiques de recherche par grille (*Grid Search*) ou aléatoire (*Random Search*) deviennent inefficaces.

Dans ce projet, nous implémentons de A à Z :

1. **Un Processus Gaussien (GP) 1D sans bruit** : Un modèle probabiliste de régression non paramétrique permettant de modéliser une fonction inconnue et d'estimer son incertitude.
2. **L'Optimisation Bayésienne (BO)** : Un algorithme qui utilise le Processus Gaussien comme substitut (*surrogate model*) et une fonction d'acquisition (*Expected Improvement*) pour déterminer intelligemment le prochain point à évaluer.
3. **L'utilisation du framework `GPyOpt**` : L'application des outils standards de l'industrie pour résoudre des problèmes d'optimisation complexes.

## 🎓 Objectifs d'apprentissage

* **Processus Gaussien (GP)** : Comprendre la distribution sur les fonctions, la fonction de moyenne et les fonctions de noyau (*Kernel / Covariance*).
* **Noyau RBF (Radial Basis Function)** : Calculer la matrice de covariance basée sur la distance euclidienne exponentiée entre les points.
* **Régression par Processus Gaussien (Kriging)** : Prédire la moyenne $\mu_*$ et la variance $\sigma_*^2$ pour de nouvelles entrées $X_*$.
* **Fonctions d'Acquisition** : Manipuler la notion d'*Expected Improvement* (EI) pour équilibrer l'exploration (zones à forte incertitude) et l'exploitation (zones à forte performance).
* **Frameworks GPy & GPyOpt** : Automatiser l'optimisation bayésienne à l'aide de bibliothèques Python spécialisées.

## 🛠️ Spécifications techniques

* **OS** : Ubuntu 20.04 LTS | **Python** : 3.9
* **Dépendances** :
* `numpy` (1.25.2)
* `GPy`
* `gpyopt`


* **Style** : Respect strict des règles de la norme `pycodestyle` (version 2.11.1).
* **Documentation** : *Docstrings* obligatoires pour tous les modules, classes et fonctions.

---

## 📂 Architecture des Tâches

| # | Tâche | Fichier | Description |
| --- | --- | --- | --- |
| **0** | **Initialize Gaussian Process** | `0-gp.py` | Classe `GaussianProcess` avec constructeur et calcul du noyau RBF. |
| **1** | **Gaussian Process Prediction** | `1-gp.py` | Méthode `predict(X_s)` calculant la moyenne et la variance prédictives. |
| **2** | **Update Gaussian Process** | `2-gp.py` | Méthode `update(X_new, Y_new)` pour mettre à jour l'échantillon et la matrice de covariance $K$. |
| **3** | **Initialize Bayesian Optimization** | `3-bayes_opt.py` | Classe `BayesianOptimization` initialisant l'espace de recherche et le GP. |
| **4** | **Bayesian Optimization - Acquisition** | `4-bayes_opt.py` | Calcul de la fonction d'acquisition *Expected Improvement* (EI). |
| **5** | **Bayesian Optimization** | `5-bayes_opt.py` | Boucle d'optimisation bayésienne complète. |
| **6** | **Bayesian Optimization with GPyOpt** | `6-gpyopt.py` | Résolution du problème d'optimisation via la bibliothèque `GPyOpt`. |

---

## 🔬 Focus Mathématique : Noyau RBF (Tâche 0)

La fonction de covariance (noyau RBF ou Gaussien) entre deux jeux de points $X_1$ (de taille $m \times 1$) et $X_2$ (de taille $n \times 1$) est définie par la formule :

$$K(X_1, X_2) = \sigma_f^2 \exp\left( -\frac{1}{2 l^2} d(X_1, X_2)^2 \right)$$

Où :

* **$\sigma_f$** (`sigma_f`) : L'écart-type d'amplitude contrôlant la variance verticale de la fonction.
* **$l$** (`l`) : La longueur d'échelle (*length-scale*) lissant la fonction horizontalement.
* **$d(X_1, X_2)^2$** : La matrice des distances euclidiennes au carré entre chaque paire de points $(x_1 \in X_1, x_2 \in X_2)$.

---

## 🛠️ Validation de la Tâche 0 : Initialisation du GP

Pour tester l'instanciation de la classe et le calcul de la matrice de covariance $K$ :

```bash
chmod +x 0-main.py
./0-main.py

```

### Exemple de sortie :

```text
True
True
0.6
2
(2, 2) [[4.         0.13150595]
 [0.13150595 4.        ]]
True

```

## ✍️ Auteur

* **Mathieu** - *Étudiant en apprentissage Machine Learning* - [Profil GitHub](https://www.google.com/search?q=https://github.com/Mathieu7483)
