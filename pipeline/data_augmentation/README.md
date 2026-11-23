<p align="center"\>
<img src="https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/data_augmentation/Data%20augmentation.png"\>
</p>

# Pipeline — Data Augmentation

## Description

Ce projet porte sur l'augmentation de données (*Data Augmentation*) appliquée aux images à l'aide de **TensorFlow** (`tf.image`) et **TensorFlow Datasets**. L'objectif est de diversifier artificiellement les jeux d'entraînement afin de limiter le surapprentissage (*overfitting*) et d'améliorer la capacité de généralisation des modèles de vision par ordinateur, notamment lorsque les données disponibles sont limitées ou déséquilibrées.

À travers une série de fonctions unitaires opérant sur des tenseurs 3D (`tf.Tensor`), ce projet couvre les transformations géométriques et photométriques fondamentales : retournement horizontal, recadrage aléatoire, rotation de 90°, ajustement du contraste, variation de luminosité et décalage de teinte.

---

## 🛠️ Spécifications Techniques

* **OS :** Ubuntu 20.04 LTS
* **Langage :** Python 3.9
* **Bibliothèques principales :**
* `TensorFlow` (version 2.15) — seul import autorisé sauf indication contraire (`import tensorflow as tf`)
* `NumPy` (version 1.25.2)
* `TensorFlow Datasets` (version 4.9.2) (`pip install --user tensorflow-datasets==4.9.2`)


* **Style de code :** Conformité stricte à la norme `pycodestyle` (version 2.11.1).
* **Exécutabilité :** Tous les fichiers doivent se terminer par une nouvelle ligne, commencer par `#!/usr/bin/env python3` et être exécutables (`chmod +x`).
* **Documentation :** Documentation obligatoire pour tous les modules, classes et fonctions.

---

## 📚 Concepts Clés

* **Data Augmentation :** Génération de nouvelles variantes d'exemples d'entraînement par transformations préservant les étiquettes, réduisant l'overfitting et renforçant l'invariance aux conditions réelles.
* **Transformations géométriques :** Modifications spatiales de l'image (retournement horizontal via `tf.image.flip_left_right`, recadrage aléatoire via `tf.image.random_crop`, rotation par multiples de 90° via `tf.image.rot90`).
* **Transformations photométriques :** Variations de l'apparence colorimétrique et lumineuse (contraste via `tf.image.random_contrast`, luminosité via `tf.image.random_brightness`, teinte via `tf.image.random_hue`).
* **Augmentation automatisée :** Utilisation d'approches guidées par le ML (ex. *AutoAugment*, *RandAugment*, recherche d'hyperparamètres ou apprentissage par renforcement) pour découvrir des politiques de transformation adaptées à une tâche donnée.

---

## 📂 Architecture des Tâches et Liens GitHub

Le tableau ci-dessous présente les tâches du projet ainsi que leurs liens directs vers les fichiers sources dans le dépôt GitHub.

| # | Tâche | Fichier Source | Description |
| --- | --- | --- | --- |
| **0** | **Flip** | [`0-flip.py`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/data_augmentation/0-flip.py) | Retourne horizontalement une image représentée par un `tf.Tensor` 3D. |
| **1** | **Crop** | [`1-crop.py`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/data_augmentation/1-crop.py) | Effectue un recadrage aléatoire d'une image 3D selon une taille cible donnée. |
| **2** | **Rotate** | [`2-rotate.py`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/data_augmentation/2-rotate.py) | Fait pivoter une image 3D d'un multiple de 90 degrés dans le sens antihoraire. |
| **3** | **Contrast** | [`3-contrast.py`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/data_augmentation/3-contrast.py) | Ajuste aléatoirement le contraste d'une image 3D dans un intervalle spécifié. |
| **4** | **Brightness** | [`4-brightness.py`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/data_augmentation/4-brightness.py) | Modifie aléatoirement la luminosité d'une image 3D selon un facteur maximal donné. |
| **5** | **Hue** | [`5-hue.py`](https://www.google.com/search?q=https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/pipeline/data_augmentation/5-hue.py) | Applique un décalage aléatoire de teinte (*hue*) sur une image 3D. |

---

## ⚙️ Installation & Utilisation

### 1. Installation des dépendances

```bash
pip install --user tensorflow-datasets==4.9.2

```

### 2. Exécution d'un script de test (Exemple Task 0)

```bash
cd holbertonschool-machine_learning/pipeline/data_augmentation
chmod +x 0-flip.py 0-main.py
./0-main.py

```

### 3. Vérification de la conformité et de la documentation

```bash
pycodestyle *.py
python3 -c 'print(__import__("0-flip").flip_image.__doc__)'

```

---

## ✍️ Auteur

* **Mathieu** — *Machine Learning Student @ Holberton School* — [Profil GitHub](https://github.com/Mathieu7483)