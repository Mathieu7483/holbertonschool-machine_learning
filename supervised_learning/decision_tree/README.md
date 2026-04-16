<p align="center"\>
<img src="https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/supervised_learning/decision_tree/a-high-end-technical-illustration-showing-the-evol.png"\>
</p>

# Decision Trees & Random Forests

## 🌳 Description

Ce projet porte sur l'implémentation "from scratch" d'algorithmes d'apprentissage supervisé et non supervisé basés sur les arbres. L'objectif est de comprendre en profondeur la structure des **Arbres de Décision**, leur agrégation en **Random Forests**, et leur variante pour la détection d'anomalies : les **Isolation Forests**.

Plutôt que d'utiliser *Scikit-Learn*, nous avons construit ici chaque brique : de la récursion pour calculer la profondeur, au choix du meilleur split via l'indice de **Gini**, jusqu'à la gestion des variables continues.

## 🎓 Objectifs d'apprentissage

  * **Structure de données** : Manipulation d'arbres binaires (Nodes & Leaves) via l'orienté objet.
  * **Algorithmes de prédiction** : Implémentation des méthodes `fit` et `predict`.
  * **Critères de division** : Compréhension de l'impureté de Gini et de l'Entropie.
  * **Ensemble Learning** : Création de Random Forests pour réduire la variance et améliorer la précision.
  * **Détection d'anomalies** : Utilisation des *Isolation Random Forests* pour identifier des points atypiques (outliers) basés sur la profondeur moyenne des feuilles.

## 🛠️ Spécifications techniques

  * **OS** : Ubuntu 20.04 LTS.
  * **Langage** : Python 3.9.
  * **Bibliothèque** : NumPy (version 1.25.2).
  * **Style** : Respect strict de `pycodestyle` (version 2.11.1).

## 📂 Liste des Tâches

| \# | Tâche | Fichier | Description |
| :--- | :--- | :--- | :--- |
| **0** | **Depth** | `0-cp_depth.py` | Calcul récursif de la profondeur maximale de l'arbre. |
| **1** | **Nodes/Leaves** | `1-count_nodes.py` | Comptage du nombre total de nœuds et de feuilles. |
| **2** | **Print Tree** | `2-print_tree.py` | Visualisation textuelle de la structure de l'arbre. |
| **3** | **Get Leaves** | `3-get_leaves.py` | Récupération de la liste de toutes les feuilles. |
| **4** | **Bounds** | `4-update_bounds.py` | Gestion des seuils (thresholds) pour les variables continues. |
| **5** | **Indicator** | `5-update_indicator.py` | Détermination des individus appartenant à un nœud spécifique. |
| **6** | **Predict** | `6-predict.py` | Implémentation de la fonction de prédiction finale. |
| **7** | **Fit** | `7-build_tree.py` | Entraînement de l'arbre par division récursive. |
| **8** | **Gini** | `8-gini_split.py` | Utilisation de l'impureté de Gini comme critère de split. |
| **9** | **Random Forests** | `9-random_forest.py` | Agrégation de plusieurs arbres de décision. |
| **10** | **IRF 1** | `10-isolation_tree.py` | Arbres d'isolation pour la détection d'outliers. |
| **11** | **IRF 2** | `11-isolation_forest.py` | Forêt d'isolation et détection des suspects (anomalies). |

-----

## 🔍 Focus Technique : Isolation Forest

Contrairement aux arbres classiques qui cherchent à classer, l'**Isolation Forest** cherche à isoler. L'idée est simple mais géniale : une anomalie est plus facile à isoler qu'un point normal. Elle se retrouvera donc à une profondeur beaucoup plus faible dans l'arbre.

  * Plus la profondeur moyenne (`mean depth`) est petite, plus l'individu est considéré comme un **suspect**.

## 🚀 Utilisation

Pour tester la détection d'anomalies (Task 11) :

```bash
./11-main.py
```

Cela générera un fichier `bassins2.png` montrant les zones de densité et identifiera les points suspects basés sur leur profondeur dans les arbres d'isolation.

## ✍️ Auteur

  * **Mathieu** - *Étudiant en programmation (41 ans)* - [GitHub Profile](https://www.google.com/search?q=https://github.com/Mathieu7483)

-----

### 👨‍🏫 Le mot du Prof

Mathieu, la tâche 11 est cruciale. Ta méthode `suspects` doit trier les individus par profondeur croissante. C'est le moment où tu passes de la théorie à la pratique : identifier concrètement des données aberrantes dans un dataset.

**Maintenant que tu as dompté les arbres, es-tu prêt à explorer les réseaux de neurones (Classification Using Neural Networks) pour ton prochain projet ?**
