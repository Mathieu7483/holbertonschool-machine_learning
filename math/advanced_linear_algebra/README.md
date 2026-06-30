<p align="center"\>
<img src="https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/math/advanced_linear_algebra/a-realistic--cinematic-photograph-of-a-high-tech-m.png"\>
</p>

---

# Advanced Linear Algebra

## 📐 Description

Ce projet est dédié à l'implémentation bas niveau et purement algorithmique d'opérations matricielles avancées essentielles au Machine Learning et à la science des données. L'objectif est de reconstruire, à partir de listes de listes Python et sans utiliser de bibliothèques externes, les blocs fondamentaux permettant de calculer l'inverse d'une matrice carrée (via les mineurs, cofacteurs et l'adjugée) ainsi que d'analyser la définiteness (*definiteness*) d'une matrice.

## 🎓 Objectifs d'apprentissage

* **Déterminant** : Comprendre sa signification géométrique (facteur d'échelle de volume) et maîtriser son calcul par récursion (formule de Leibniz / expansion de Laplace).
* **Inversion de Matrice** : Décomposer le pipeline d'inversion analytique : Matrice $\to$ Mineurs $\to$ Cofacteurs $\to$ Adjugée $\to$ Inverse.
* **Matrices Singulières** : Identifier mathématiquement les matrices non inversibles (déterminant nul).
* **Definiteness** : Déterminer la nature d'une matrice (définie positive, semi-définie positive, etc.) en exploitant ses valeurs propres (*eigenvalues*) et ses mineurs principaux.

## 🛠️ Spécifications techniques

* **OS** : Ubuntu 20.04 LTS | **Python** : 3.9
* **Style** : Conformité réglementaire stricte avec la norme `pycodestyle` (2.11.1).
* **Restrictions** : **Aucun import de module externe autorisé** dans vos fichiers sources (interdiction d'utiliser `numpy` pour effectuer les calculs à votre place).

---

## 📂 Organisation du Répertoire et des Tâches

Le projet suit une progression logique où chaque tâche réutilise le code de la précédente pour bâtir un pipeline d'algèbre complet :

| # | Tâche | Fichier | Description |
| --- | --- | --- | --- |
| **0** | **Determinant** | `0-determinant.py` | Calcule le déterminant d'une matrice de taille $n \times n$ par récursion. Gère le cas limite de la matrice vide $[[]]$ ($0 \times 0$). |
| **1** | **Minor** | `1-minor.py` | Génère la matrice des mineurs en calculant le déterminant des sous-matrices associées. |
| **2** | **Cofactor** | `2-cofactor.py` | Applique l'échiquier de signes $(-1)^{i+j}$ sur la matrice des mineurs. |
| **3** | **Adjugate** | `3-adjugate.py` | Transpose la matrice des cofacteurs pour obtenir la matrice adjointe (ou adjugée). |
| **4** | **Inverse** | `4-inverse.py` | Calcule l'inverse final en divisant la matrice adjugée par le déterminant : $A^{-1} = \frac{1}{\det(A)} \cdot \text{adj}(A)$. |
| **5** | **Definiteness** | `5-definiteness.py` | Identifie le type de définition d'une matrice (Positive definite, Positive semi-definite, Negative definite, Negative semi-definite, ou Indefinite). |

---

## 🔬 Focus Algorithmique : Le Pipeline d'Inversion Analytique

Pour inverser une matrice $A$ sans utiliser l'élimination de Gauss-Jordan, on implémente la méthode des cofacteurs, structurée comme suit :

1. **Calcul du Déterminant ($\det(A)$)** : Si $\det(A) == 0$, la matrice est singulière et ne peut pas être inversée.
2. **Matrice des Mineurs ($M$)** : Pour chaque élément $a_{ij}$, on calcule le déterminant de la sous-matrice obtenue en supprimant la ligne $i$ et la colonne $j$.
3. **Matrice des Cofacteurs ($C$)** : On ajuste le signe de chaque élément de $M$ : $C_{ij} = (-1)^{i+j} \cdot M_{ij}$.
4. **Matrice Adjugée ($\text{adj}(A)$)** : On transpose la matrice des cofacteurs : $\text{adj}(A) = C^T$.
5. **Matrice Inverse ($A^{-1}$)** : On applique la formule finale.

---

## 🛠️ Validation de la Tâche 0 : Déterminant

Le script de validation teste la robustesse du code face aux cas limites et aux matrices non valides (non carrées ou mauvais types) :

```bash
chmod +x 0-main.py
./0-main.py

```

### Sortie attendue :

```text
1
5
-2
0
-192
matrix must be a list of lists
matrix must be a square matrix

```

## ✍️ Auteur

  * **Mathieu** - *Programming student, specialization Machine Learning* - [👤 My Github profile](https://github.com/Mathieu7483)