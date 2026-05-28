# Deep Convolutional Architectures

## 🏢 Description

Ce projet se concentre sur l'étude et le développement d'architectures de réseaux de neurones convolutifs profonds. Au fur et à mesure que les réseaux s'approfondissent, ils se heurtent au problème majeur de la disparition ou de l'explosion du gradient (*Vanishing/Exploding Gradient*). L'objectif ici est d'implémenter les blocs fondamentaux et les structures complètes des modèles phares de l'industrie : **GoogLeNet**, **ResNet**, **ResNeXt**, et **DenseNet**, en s'appuyant sur l'API fonctionnelle de Keras.

## 🎓 Objectifs d'apprentissage

* **Lecture de recherche** : Savoir répliquer une architecture de réseau complexe en interprétant les tableaux et descriptions issus de publications scientifiques.
* **Skip Connections (Connexions résiduelles)** : Comprendre comment les raccourcis permettent d'entraîner des réseaux contenant plus de 100 couches sans dégrader les performances.
* **Couches Bottleneck** : Utiliser des convolutions $1 \times 1$ pour réduire la dimensionnalité et optimiser les coûts de calcul.
* **Concepts Avancés** : Maîtriser le fonctionnement des blocs Inception, des transformations agrégées (ResNeXt) et des connexions denses (DenseNet).

## 🛠️ Spécifications techniques

* **OS** : Ubuntu 20.04 LTS | **Python** : 3.9
* **Frameworks** : NumPy (1.25.2) et TensorFlow / Keras (2.15).
* **Style** : Conformité réglementaire avec la norme `pycodestyle` (2.11.1).
* **Contrainte d'importation** : Sauf mention contraire, seul l'import `from tensorflow import keras as K` est autorisé.

---

## 📂 Liste des Tâches (Implementation Focus)

| # | Tâche | Fichier | Description |
| --- | --- | --- | --- |
| **0** | **Identity Block** | `0-identity_block.py` | Construit un bloc d'identité résiduel (sans modification de dimensions) avec couches *bottleneck*. |
| **1** | **Projection Block** | `1-projection_block.py` | Construit un bloc de projection résiduel avec une convolution $1 \times 1$ sur le raccourci pour adapter les dimensions. |
| **2** | **ResNet-50** | `2-resnet50.py` | Assemblage complet de l'architecture ResNet-50 d'après la publication de 2015. |

---

## 🔬 Focus Architectural : Le Bloc d'Identité (ResNet)

L'architecture **ResNet** introduit des blocs où l'entrée $X$ est directement ajoutée à la sortie des transformations du bloc, calculant ainsi une fonction résiduelle $F(X) + X$.

Le bloc d'identité utilise une structure dite **Bottleneck** découpée en 3 étapes :

1. Une convolution $1 \times 1$ pour réduire le nombre de canaux (économie de calcul).
2. Une convolution $3 \times 3$ pour capturer les caractéristiques spatiales.
3. Une convolution $1 \times 1$ pour restaurer la dimension initiale des canaux avant l'addition de l'identité.

> ⚠️ **Règle stricte du projet** : Chaque couche de convolution doit être immédiatement suivie d'une normalisation par lots (*Batch Normalization*) sur l'axe des canaux, puis d'une activation ReLU. L'identité est ajoutée **juste avant** la toute dernière activation ReLU du bloc.

---

## 🛠️ Validation du modèle

Un script de test comme `0-main.py` permet de valider le comportement et d'afficher le résumé structurel à l'aide de la méthode de Keras :

```bash
chmod +x 0-main.py
./0-main.py

```

### Extrait de la topologie attendue (`model.summary()`) :

```text
==================================================================================================
 Layer (type)                Output Shape                 Param #   Connected to                  
==================================================================================================
 input_1 (InputLayer)        [(None, 224, 224, 256)]      0         []                            
 conv2d (Conv2D)             (None, 224, 224, 64)         16448     ['input_1[0][0]']             
 batch_normalization (...)   (None, 224, 224, 64)         256       ['conv2d[0][0]']              
 activation (Activation)     (None, 224, 224, 64)         0         ['batch_normalization[0][0]'] 
 ...
 add (Add)                   (None, 224, 224, 256)        0         ['batch_normalization_2[0][0]'
                                                                     'input_1[0][0]']            
 activation_2 (Activation)   (None, 224, 224, 256)        0         ['add[0][0]']                 
==================================================================================================

```

## ✍️ Auteur

  * **Mathieu** - *Programming student, specialization Machine Learning* - [👤 My Github profile](https://github.com/Mathieu7483)