<p align="center"\>
<img src="https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/supervised_learning/error_analysis/an-abstract-representation-of-mathematical-formula.png"\>
</p>

# Error Analysis in Machine Learning

## 📉 Description

Ce projet se concentre sur l'évaluation quantitative et qualitative des modèles de classification. L'objectif est de transformer des prédictions brutes en métriques exploitables (Précision, Rappel, F1-Score) à l'aide d'une **matrice de confusion**. On y explore également les concepts théoriques du compromis **biais-variance** et l'erreur de Bayes pour comprendre les limites fondamentales de nos modèles.

## 🎓 Objectifs d'apprentissage

* **Métriques de Performance** : Calculer et interpréter la Sensibilité (Rappel), la Spécificité et la Précision.
* **Matrice de Confusion** : Construire et lire une matrice pour identifier quelles classes sont confondues.
* **Théorie de l'Erreur** : Distinguer l'erreur évitable (biais) de l'erreur irréductible (Bayes).
* **Diagnostic** : Savoir si un modèle souffre d'un sous-apprentissage (High Bias) ou d'un sur-apprentissage (High Variance).

## 🛠️ Spécifications techniques

* **OS** : Ubuntu 20.04 LTS | **Python** : 3.9.
* **Bibliothèque** : NumPy (1.25.2) exclusivement.
* **Style** : Conformité totale avec `pycodestyle` (2.11.1).
* **Contrainte** : Pas de bibliothèques de haut niveau comme Scikit-learn ; tout doit être calculé via l'algèbre matricielle avec NumPy.

---

## 📂 Liste des Tâches (Implementation)

| # | Tâche | Fichier | Description |
| --- | --- | --- | --- |
| **0** | **Create Confusion** | `0-create_confusion.py` | Génère une matrice de confusion à partir de labels One-Hot. |
| **1** | **Sensitivity** | `1-sensitivity.py` | Calcule le rappel (TPR) pour chaque classe de la matrice. |
| **2** | **Precision** | `2-precision.py` | Calcule la valeur prédictive positive pour chaque classe. |
| **3** | **Specificity** | `3-specificity.py` | Calcule le taux de vrais négatifs (TNR) par classe. |
| **4** | **F1 Score** | `4-f1_score.py` | Calcule la moyenne harmonique de la précision et du rappel. |

## 🧠 Tâches Théoriques (Analyse)

| # | Tâche | Fichier | Description |
| --- | --- | --- | --- |
| **5** | **Dealing with Error** | `5-error_handling` | QCM/Analyse sur la gestion du biais et de la variance. |
| **6** | **Compare and Contrast** | `6-compare_contrast` | Analyse des différences entre les métriques d'évaluation. |

---

## 🔬 Focus : La Matrice de Confusion

La matrice de confusion est un outil où les lignes représentent les classes réelles et les colonnes les classes prédites.

* Les éléments de la **diagonale** sont les prédictions correctes.
* Les éléments **hors diagonale** révèlent précisément quel chiffre (dans le cas de MNIST) le modèle confond avec un autre (par exemple, confondre un `4` avec un `9`).

## ✍️ Author

  * **Mathieu** - *Programming student, specialization Machine Learning* - [👤 My Github profile](https://github.com/Mathieu7483)

