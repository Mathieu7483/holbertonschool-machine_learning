# Transfer Learning — CIFAR-10 Classification

## 🧠 Description

Ce projet met en œuvre les principes du **Transfer Learning** (apprentissage par transfert) et du **Fine-Tuning**. L'objectif est d'utiliser un modèle d'état de l'art pré-entraîné sur le dataset géant ImageNet (via `tf.keras.applications`) et de l'adapter pour classifier avec précision le dataset **CIFAR-10**. Ce processus démontre comment capitaliser sur des caractéristiques visuelles complexes déjà apprises (bords, textures, formes géométriques) pour obtenir d'excellentes performances sur un jeu de données restreint sans nécessiter des ressources de calcul massives.

## 🎓 Objectifs d'apprentissage

* **Concept de Transfer Learning** : Comprendre comment réutiliser les couches de caractéristiques d'un modèle pré-entraîné.
* **Fine-Tuning & Freezing** : Maîtriser le gel (*freezing*) des couches pour conserver les poids appris et n'entraîner que le nouveau classifieur (couches Top / Fully Connected).
* **Adaptation de Tenseurs** : Redimensionner dynamiquement les entrées de faible résolution ($32 \times 32$) pour correspondre aux attentes des architectures lourdes via des couches `Lambda`.
* **Pratique Industrielle** : Atteindre un objectif de performance exigeant ($\ge 87\%$ d'exactitude sur le jeu de validation).

## 🛠️ Spécifications techniques

* **OS** : Ubuntu 20.04 LTS | **Python** : 3.9
* **Frameworks** : NumPy (1.25.2) et TensorFlow / Keras (2.15).
* **Style** : Respect strict des règles d'écriture de la norme `pycodestyle` (2.11.1).
* **Contrainte d'importation** : Seul l'import `from tensorflow import keras as K` est autorisé.

---

## 📂 Liste des Tâches

| # | Tâche | Fichier | Description |
| --- | --- | --- | --- |
| **0** | **Transfer Knowledge** | `0-transfer.py` | Script Python complet d'entraînement de CNN avec Keras Applications + fonction de prétraitement des données. |
| **1** | **Blog Post** | `N/A` | Rédaction d'un rapport/article technique documentant les expérimentations, architectures testées et hyperparamètres choisis. |

---

## 🔬 Stratégie d'Architecture et Optimisation

Pour adapter une application Keras (conçue pour des images de $224 \times 224$ pixels) aux images de taille $32 \times 32$ de CIFAR-10, la structure du réseau doit suivre ce pipeline :

1. **Couche de Prétraitement** : Fonction spécifique à l'application choisie (ex: `K.applications.resnet50.preprocess_input`).
2. **Couche Lambda Keras** : Redimensionnement à la volée des images (ex: $32 \times 32 \to 224 \times 224$) avec `tf.image.resize`.
3. **Base Pré-entraînée (Frozen)** : Modèle Keras importé avec `include_top=False` dont l'ensemble des couches ont leur propriété `trainable` configurée à `False`.
4. **Tête de Classification (Trainable)** : Ajout d'une couche de Global Average Pooling, suivie de couches denses (`Dense`) avec Dropout pour éviter l'overfitting, menant à la couche Softmax à 10 sorties.

---

## 🛠️ Exécution et Validation

Le modèle final doit être sauvegardé sous le nom de `cifar10.h5` à la racine du répertoire. Un script de test permet de charger le modèle et d'évaluer ses performances sur le jeu de test :

```bash
chmod +x 0-main.py
./0-main.py

```

### Métriques attendues :

* **Validation Accuracy** : $\ge 87\%$
* L'évaluation ne doit pas lever d'erreurs d'échelles ou de dimensions de tenseurs.

## ✍️ Auteur

* **Mathieu** - *Étudiant en apprentissage Machine Learning (42 ans)* - [GitHub Profile](https://www.google.com/search?q=https://github.com/Mathieu7483)