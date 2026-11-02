<p align="center"\>
<img src="https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/pandas/Pandas.png"\>
</p>

# Pipeline — Pandas

## Description

Ce projet aborde l'utilisation de la bibliothèque **Pandas** en Python, un outil incontournable pour la manipulation, le nettoyage, l'analyse et la visualisation de données structurées.

À travers une série d'exercices pratiques, ce projet couvre l'ensemble des opérations fondamentales sur les structures de données `pd.Series` et `pd.DataFrame` : chargement depuis divers formats de fichiers, indexation, découpage (*slicing*), manipulation de colonnes, fusion (*merge/concat*), calculs statistiques et visualisations graphiques.

---

## 🛠️ Spécifications Techniques

* **OS :** Ubuntu 20.04 LTS
* **Langage :** Python 3.9
* **Bibliothèques principales :**
* `Pandas` (version 2.2.2) (`pip install --user pandas==2.2.2`)
* `NumPy` (version 1.25.2)


* **Style de code :** Conformité stricte à la norme `pycodestyle` (version 2.11.1).
* **Exécutabilité :** Tous les scripts doivent se terminer par une nouvelle ligne, commencer par le shebang `#!/usr/bin/env python3` et être exécutables (`chmod +x`).
* **Documentation :** Documentation obligatoire pour tous les modules, classes et fonctions.

---

## 📚 Concepts Clés

* **`pd.Series` & `pd.DataFrame` :** Structures de données fondamentales unidimensionnelles et bidimensionnelles étiquetées.
* **Indexation et Slicing :** Accès aux données par étiquette (`.loc`), par position entière (`.iloc`) ou par indexation booléenne.
* **Indexation hiérarchique (*MultiIndex*) :** Manipulation de données multi-niveaux.
* **Agrégation et Fusion :** Combinaison de DataFrames via `concat`, `merge` ou `join`, et regroupement via `groupby`.
* **Analyse statistique :** Utilisation des méthodes descriptives (`describe`, `mean`, `std`, etc.) et traitement des valeurs manquantes (`fillna`).

---

## 📂 Architecture des Tâches et Liens GitHub

Le tableau ci-dessous répertorie l'ensemble des tâches du projet ainsi que les liens directs vers les fichiers sources dans le dépôt GitHub.

| # | Tâche | Fichier Source | Description |
| --- | --- | --- | --- |
| **0** | **From Numpy** | [`0-from_numpy.py`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/pandas/0-from_numpy.py) | Crée un DataFrame Pandas à partir d'un ndarray NumPy avec des colonnes alphabétiques capitalisées. |
| **1** | **From Dictionary** | [`1-from_dictionary.py`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/pandas/1-from_dictionary.py) | Crée un DataFrame Pandas à partir d'un dictionnaire de données. |
| **2** | **From File** | [`2-from_file.py`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/pandas/2-from_file.py) | Charge des données tabulaires à partir d'un fichier CSV ou Excel. |
| **3** | **Rename** | [`3-rename.py`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/pandas/3-rename.py) | Renomme des colonnes et modifie les types de données dans un DataFrame. |
| **4** | **To Numpy** | [`4-to_numpy.py`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/pandas/4-to_numpy.py) | Convertit un DataFrame Pandas en tableau NumPy brut. |
| **5** | **Slice** | [`5-slice.py`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/pandas/5-slice.py) | Effectue des découpages spécifiques sur les lignes et colonnes d'un DataFrame. |
| **6** | **Flip it and Switch it** | [`6-flip.py`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/pandas/6-flip.py) | Réordonne, transpose et pivote les axes d'un DataFrame. |
| **7** | **Sort** | [`7-sort.py`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/pandas/7-sort.py) | Trie un DataFrame selon des colonnes et des critères spécifiques. |
| **8** | **Prune** | [`8-prune.py`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/pandas/8-prune.py) | Supprime les lignes ou colonnes contenant des valeurs manquantes. |
| **9** | **Fill** | [`9-fill.py`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/pandas/9-fill.py) | Remplace ou impute les valeurs manquantes (`NaN`) selon des règles précises. |
| **10** | **Indexing** | [`10-indexing.py`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/pandas/10-indexing.py) | Sélectionne et filtre des sous-ensembles de données par indexation avancée. |
| **11** | **Concat** | [`11-concat.py`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/pandas/11-concat.py) | Concatène plusieurs DataFrames le long d'un axe spécifique. |
| **12** | **Hierarchy** | [`12-hierarchy.py`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/pandas/12-hierarchy.py) | Met en place et manipule une indexation hiérarchique (*MultiIndex*). |
| **13** | **Analyze** | [`13-analyze.py`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/pandas/13-analyze.py) | Calcule des statistiques descriptives et analyse les corrélations. |
| **14** | **Visualize** | [`14-visualize.py`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/pandas/14-visualize.py) | Génère des visualisations graphiques à partir des données du DataFrame. |

---

## ⚙️ Installation & Utilisation

### 1. Installation de l'environnement

```bash
pip install --user pandas==2.2.2 numpy==1.25.2

```

### 2. Exécution d'un script de test (Exemple Task 0)

```bash
chmod +x 0-from_numpy.py 0-main.py
./0-main.py

```

---

## ✍️ Auteur

* **Mathieu** — *Machine Learning Student @ Holberton School* — [Profil GitHub](https://github.com/Mathieu7483)

---