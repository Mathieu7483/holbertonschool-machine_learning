# Regularization: Preventing Overfitting

## 🛡️ Description

L'objectif de ce projet est de maîtriser les différentes techniques de régularisation pour améliorer la capacité de généralisation des réseaux de neurones. Un modèle performant sur ses données d'entraînement mais médiocre sur de nouvelles données est inutile ; la régularisation permet de limiter la complexité du modèle pour le rendre plus robuste. Nous explorerons les méthodes mathématiques (**L2 Regularization**) et stochastiques (**Dropout**), ainsi que les stratégies d'entraînement comme l'**Early Stopping**.

## 🎓 Objectifs d'apprentissage

* **Overfitting** : Comprendre pourquoi un modèle sur-apprend et comment le détecter.
* **Régularisation L2** : Implémenter la "Weight Decay" pour pénaliser les poids trop élevés.
* **Dropout** : Simuler un réseau plus robuste en désactivant aléatoirement des neurones pendant l'entraînement.
* **Early Stopping** : Arrêter l'entraînement au moment optimal pour éviter que le coût de validation ne remonte.
* **Data Augmentation** : Créer artificiellement de la donnée pour améliorer la robustesse.

## 🛠️ Spécifications techniques

* **OS** : Ubuntu 20.04 LTS | **Python** : 3.9.
* **Bibliothèques** : NumPy (1.25.2) et TensorFlow (2.15).
* **Style** : Respect strict de `pycodestyle` (2.11.1).
* **Initialisation** : Utilisation impérative de `tf.keras.initializers.VarianceScaling`.

---

## 📂 Liste des Tâches (Numpy & Math)

| # | Tâche | Fichier | Description |
| --- | --- | --- | --- |
| **0** | **L2 Reg Cost** | `0-l2_reg_cost.py` | Calcul du coût incluant la pénalité L2. |
| **1** | **L2 Gradient Descent** | `1-l2_reg_gradient_descent.py` | Mise à jour des poids avec régularisation L2. |
| **4** | **Forward Prop Dropout** | `4-dropout_forward_prop.py` | Propagation avant avec masque de Dropout. |
| **5** | **Dropout Grad Descent** | `5-dropout_gradient_descent.py` | Backpropagation avec masque de Dropout. |

## 🚀 Tâches Avancées (TensorFlow & Callbacks)

| # | Tâche | Fichier | Description |
| --- | --- | --- | --- |
| **2** | **L2 Cost TF** | `2-l2_reg_cost.py` | Calcul du coût L2 via les outils de TensorFlow. |
| **3** | **L2 Layer TF** | `3-l2_reg_create_layer.py` | Création d'une couche Dense intégrant la régularisation L2. |
| **6** | **Dropout Layer TF** | `6-dropout_create_layer.py` | Création d'une couche Dense intégrant le Dropout. |
| **7** | **Early Stopping** | `7-early_stopping.py` | Définition de la fonction d'arrêt précoce. |
| **8** | **Blog Post** | `N/A` | Synthèse sur l'importance de la régularisation. |

---

## 🔬 Focus : L2 vs Dropout

* **L2 Regularization** : Ajoute une pénalité à la fonction de coût proportionnelle au carré des poids. Cela force les poids à rester petits, empêchant le réseau de trop se focaliser sur des détails insignifiants du dataset.
* **Dropout** : À chaque itération, chaque neurone a une probabilité $1 - keep\_prob$ d'être ignoré. Cela force le réseau à apprendre des représentations redondantes et l'empêche de devenir dépendant de neurones spécifiques.

## ✍️ Author

  * **Mathieu** - *Programming student, specialization Machine Learning* - [👤 My Github profile](https://github.com/Mathieu7483)