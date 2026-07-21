<p align="center"\>
<img src="https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/unsupervised_learning/clustering/a-realistic--cinematic-photograph-of-a-data-scient.png"\>
</p>


# Unsupervised Learning — Clustering

## 📊 Description

Ce projet est dédié à la segmentation et à la structuration automatique de données non étiquetées par des techniques de **Clustering**. Le clustering permet de regrouper des observations similaires au sein d'un même groupe (*cluster*) tout en séparant les éléments distincts.

Dans ce dépôt, nous étudions et implémentons pas à pas :

1. **K-means Clustering** : Un algorithme de partitionnement strict (*hard clustering*) basé sur la distance euclidienne et la minimisation de la variance intra-cluster.
2. **Gaussian Mixture Models (GMM) & Algorithme EM** : Un modèle probabiliste souple (*soft clustering*) estimant l'appartenance d'un point à une distribution gaussienne via les étapes d'Espérance (*Expectation*) et de Maximisation (*Maximization*).
3. **Clustering Hiérarchique Agglomératif** : La construction d'arbres de regroupement (*dendrogrammes*) basés sur le lien de Ward.
4. **Sélection du nombre de clusters** : L'utilisation de métriques d'évaluation telles que la méthode du coude (*Elbow Method*) et le critère d'information bayésien (*BIC*).

## 🎓 Objectifs d'apprentissage

* **Hard vs Soft Clustering** : Comprendre la différence entre l'affectation binaire d'un point à un cluster et l'attribution d'une distribution de probabilité.
* **Vectorisation NumPy Avancée** : Réaliser des calculs matriciels complexes et des tirages aléatoires **sans aucune boucle** (`for`/`while`).
* **Algorithme Expectation-Maximization (EM)** :
* *Étape E* : Calcul des probabilités a posteriori (*responsabilités*).
* *Étape M* : Mise à jour des paramètres du modèle (moyennes $\mu$, matrices de covariance $\Sigma$, et poids $\pi$).


* **Évaluation & Optimisation** : Utiliser le critère BIC pour éviter le surajustement (*overfitting*) lors du choix du nombre de composantes $k$.

## 🛠️ Spécifications techniques

* **OS** : Ubuntu 20.04 LTS | **Python** : 3.9
* **Dépendances** :
* `numpy` (1.25.2)
* `scikit-learn` (1.5.0)
* `scipy` (1.11.4)


* **Style** : Respect strict des standards de la norme `pycodestyle` (version 2.11.1).
* **Documentation** : Tous les modules, classes et fonctions doivent inclure des *docstrings* conformes.

---

## 📂 Architecture des Tâches

| # | Tâche | Fichier | Description |
| --- | --- | --- | --- |
| **0** | **Initialize K-means** | `0-initialize.py` | Initialise $k$ centroïdes de manière uniforme à l'intérieur des bornes min/max du jeu de données (sans boucle). |
| **1** | **K-means** | `1-kmeans.py` | Implémente l'algorithme K-means complet jusqu'à convergence des centroïdes. |
| **2** | **Variance** | `2-variance.py` | Calcule la variance totale intra-cluster (*Inertia*). |
| **3** | **Optimize k** | `3-opt_k.py` | Recherche le $k$ optimal en utilisant la méthode du coude (*Elbow method*). |
| **4** | **Initialize GMM** | `4-initialize.py` | Initialise les paramètres d'un GMM ($\pi$, $\mu$, $\Sigma$). |
| **5** | **PDF** | `5-pdf.py` | Calcule la densité de probabilité d'une loi normale multivariée. |
| **6** | **Expectation** | `6-expectation.py` | Étape E de l'algorithme EM (calcul des responsabilités). |
| **7** | **Maximization** | `7-maximization.py` | Étape M de l'algorithme EM (mise à jour des paramètres). |
| **8** | **EM** | `8-EM.py` | Boucle d'entraînement complète de l'algorithme EM pour un GMM. |
| **9** | **BIC** | `9-BIC.py` | Sélection automatique du meilleur nombre de clusters $k$ via le critère BIC. |
| **10** | **Hello, sklearn!** | `10-kmeans.py` | Implémentation rapide de K-means avec Scikit-Learn. |
| **11** | **GMM** | `11-gmm.py` | Implémentation de GMM avec Scikit-Learn. |
| **12** | **Agglomerative** | `12-agglomerative.py` | Clustering hiérarchique agglomératif avec SciPy. |

---

## 🔬 Focus Mathématique : Initialisation des Centroïdes K-means (Tâche 0)

L'initialisation des centroïdes s'effectue par un tirage suivant une loi uniforme multivariée sur l'intervalle défini par les valeurs minimales et maximales de chaque dimension du jeu de données $X$ :

$$\text{Min}_j = \min_{i} X_{i, j} \quad \text{et} \quad \text{Max}_j = \max_{i} X_{i, j} \quad \forall j \in [0, d-1]$$

Chaque centroïde $C_m$ (pour $m \in [0, k-1]$) est tiré de façon indépendante :

$$C_{m, j} \sim U(\text{Min}_j, \text{Max}_j)$$

---

## 🛠️ Validation de la Tâche 0 : Initialisation

Pour exécuter le script de test et générer des centroïdes initialisés dans l'espace des données :

```bash
chmod +x 0-main.py
./0-main.py

```

### Exemple de sortie :

```text
[[14.54730144 13.46780434]
 [20.57098466 33.55245039]
 [ 9.55556506 51.51143281]
 [48.72458008 20.03154959]
 [25.43826106 60.35542243]]

```

## ✍️ Auteur

  * **Mathieu** - *Programming student, specialization Machine Learning* - [👤 My Github profile](https://github.com/Mathieu7483)
