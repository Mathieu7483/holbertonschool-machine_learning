# TensorFlow 2 & Keras: Deep Learning Framework

## 🤖 Description
Ce projet marque la transition entre l'implémentation théorique (from scratch) et l'utilisation de frameworks industriels. En utilisant **TensorFlow 2** et son API de haut niveau **Keras**, l'objectif est de construire, entraîner et optimiser des réseaux de neurones profonds de manière efficace. On y aborde la création de modèles (Séquentiel vs Fonctionnel), la régularisation (L2, Dropout, Batch Normalization), ainsi que la gestion avancée de l'entraînement (Callbacks, Early Stopping, Learning Rate Decay).



## 🎓 Objectifs d'apprentissage
* **Modélisation** : Savoir instancier des modèles avec l'API `Sequential` et l'API `Functional`.
* **Régularisation** : Ajouter du Dropout et de la régularisation L2 pour éviter l'overfitting.
* **Optimisation** : Configurer les optimiseurs (Adam, SGD, etc.) et les fonctions de perte.
* **Gestion du temps** : Utiliser le `EarlyStopping` pour arrêter l'entraînement au moment optimal.
* **Persistance** : Sauvegarder et charger des modèles complets ou uniquement des poids au format HDF5.
* **Performance** : Intégrer la Batch Normalization pour accélérer la convergence.

## 🛠️ Spécifications techniques
* **OS** : Ubuntu 20.04 LTS | **Python** : 3.9
* **Framework** : TensorFlow 2.15 (Keras) | **NumPy** : 1.25.2
* **Contrainte d'import** : `import tensorflow.keras as K` est la seule forme autorisée.
* **Style** : Respect de `pycodestyle` (2.11.1).

## 📂 Liste des Tâches (Foundations)

| # | Tâche | Fichier | Description |
| :--- | :--- | :--- | :--- |
| **0** | **Sequential** | `0-sequential.py` | Construction d'un modèle avec l'API `Sequential`. |
| **1** | **Input** | `1-input.py` | Construction d'un modèle avec l'API `Functional`. |
| **2** | **Optimize** | `2-optimize.py` | Configuration de l'optimiseur Adam avec decay. |
| **3** | **One Hot** | `3-one_hot.py` | Conversion de vecteurs de labels en matrices One-Hot. |
| **4** | **Train** | `4-train.py` | Implémentation du cycle d'entraînement (`fit`). |
| **5** | **Validate** | `5-validate.py` | Ajout de données de validation pendant l'entraînement. |
| **6** | **Early Stopping** | `6-early_stopping.py` | Utilisation des Callbacks pour stopper si le coût stagne. |
| **7** | **Learning Rate Decay** | `7-learning_rate_decay.py` | Réduction dynamique du taux d'apprentissage. |
| **8** | **Save Best** | `8-train.py` | Sauvegarde automatique du meilleur modèle uniquement. |

## 🚀 Tâches Avancées (Persistence & Evaluation)

| # | Tâche | Fichier | Description |
| :--- | :--- | :--- | :--- |
| **9** | **Save/Load Model** | `9-model.py` | Sauvegarde et chargement d'un modèle complet. |
| **10** | **Save/Load Weights** | `10-weights.py` | Gestion spécifique des poids du modèle. |
| **11** | **Save/Load Config** | `11-config.py` | Sauvegarde de l'architecture au format JSON. |
| **12** | **Test** | `12-test.py` | Évaluation du modèle sur un jeu de données test. |
| **13** | **Predict** | `13-predict.py` | Génération de prédictions sur de nouvelles données. |

---

## 🔬 Focus : Sequential vs Functional
Dans ce projet, tu apprendras la différence cruciale entre :
1.  **L'API Séquentielle** : Simple, pour des piles de couches linéaires.
2.  **L'API Fonctionnelle** : Flexible, permettant des graphes complexes (entrées multiples, branches partagées).



## ✍️ Auteur
* **Mathieu** - *Étudiant en programmation (41 ans)* - [GitHub Profile](https://github.com/Mathieu7483)

