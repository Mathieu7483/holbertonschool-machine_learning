<p align="center"\>
<img src="https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/supervised_learning/classification/a-clean--technical-diagram-of-a-deep-neural-networ.png"\>
</p>

# Classification Using Neural Networks

## 🧠 Description

Ce projet constitue le pilier de l'apprentissage supervisé. L'objectif est de construire entièrement, à l'aide de **NumPy**, des modèles de classification binaire et multiclasse. En partant d'un simple neurone (Régression Logistique) pour arriver à un réseau profond (**Deep L-layer Network**), ce projet couvre les mécanismes fondamentaux : propagation avant, rétropropagation du gradient, et fonctions d'activation (Sigmoïde, Tanh, ReLU, Softmax).

## 🎓 Objectifs d'apprentissage

  * **Architecture** : Maîtriser l'initialisation et la mise à jour des poids ($W$) et des biais ($b$).
  * **Vectorisation** : Optimiser les calculs matriciels pour traiter des milliers d'exemples sans boucles Python.
  * **Backpropagation** : Comprendre le calcul des gradients sur des graphes complexes.
  * **Persistance** : Sauvegarder et charger des modèles via **Pickle**.

## 🛠️ Spécifications techniques

  * **OS** : Ubuntu 20.04 LTS | **Python** : 3.9 | **NumPy** : 1.25.2.
  * **Style** : Respect de `pycodestyle` (2.11.1).
  * **Contrainte** : Interdiction d'utiliser des boucles, sauf pour les itérations d'entraînement.
  * **Multiplications** : Utilisation exclusive de `numpy.matmul`.

## 📂 Liste des Tâches (Foundations)

| \# | Tâche | Fichier | Description |
| :--- | :--- | :--- | :--- |
| **0** | **Neuron** | `0-neuron.py` | Création de la classe Neuron (Binaire). |
| **1** | **Privatize** | `1-neuron.py` | Encapsulation des attributs `W`, `b`, et `A`. |
| **2** | **Forward Prop** | `2-neuron.py` | Implémentation de la fonction Sigmoïde. |
| **3** | **Cost** | `3-neuron.py` | Calcul de la perte logistique (Log-loss). |
| **4** | **Evaluate** | `4-neuron.py` | Prédiction avec seuil à 0.5. |
| **5** | **Gradient Descent** | `5-neuron.py` | Mise à jour des paramètres du neurone. |
| **6** | **Train** | `6-neuron.py` | Première boucle d'entraînement. |
| **7** | **Upgrade Train** | `7-neuron.py` | Gestion des exceptions et itérations. |
| **8** | **NeuralNetwork** | `8-neural_network.py` | Réseau à une couche cachée. |
| **9** | **Privatize NN** | `9-neural_network.py` | Encapsulation pour NeuralNetwork. |
| **10** | **NN Forward** | `10-neural_network.py` | Propagation sur deux couches. |
| **11** | **NN Cost** | `11-neural_network.py` | Coût pour un réseau à une couche cachée. |
| **12** | **Evaluate NN** | `12-neural_network.py` | Évaluation du réseau à deux couches. |
| **13** | **NN Backprop** | `13-neural_network.py` | Descente de gradient sur deux couches. |
| **14** | **Train NN** | `14-neural_network.py` | Entraînement du NeuralNetwork. |
| **15** | **Visual Train** | `15-neural_network.py` | Ajout des options `verbose` et `graph`. |

## 🚀 Tâches Avancées (Deep Learning)

| \# | Tâche | Fichier | Description |
| :--- | :--- | :--- | :--- |
| **16** | **DeepNet** | `16-deep_neural_network.py` | Architecture à L couches (profonde). |
| **17-21** | **DNN Logic** | Divers | Forward, Cost, Evaluate et Backprop pour DNN. |
| **22-23** | **Train Deep** | `23-deep_neural_network.py` | Entraînement et monitoring du Deep Network. |
| **24** | **One-Hot Encode** | `24-one_hot_encode.py` | Encodage des labels pour le multiclasse. |
| **25** | **One-Hot Decode** | `25-one_hot_decode.py` | Conversion inverse des vecteurs One-Hot. |
| **26** | **Pickle it** | `26-persistence.py` | Sauvegarde/Chargement d'objets Python. |
| **27** | **Persistence** | `27-deep_neural_network.py` | Intégration de Pickle dans la classe DNN. |
| **28** | **Activations** | `28-deep_neural_network.py` | Support Sigmoid, Tanh, ReLU par couche. |
| **29** | **Blog Post** | `N/A` | Article technique sur la classification. |

-----

## ✍️ Auteur

  * **Mathieu** - *Programming student, specialization Machine Learning* - [👤 My Github profile](https://github.com/Mathieu7483)
