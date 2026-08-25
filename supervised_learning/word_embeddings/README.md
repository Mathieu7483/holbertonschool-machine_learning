<p align="center"\>
<img src="https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/supervised_learning/word_embeddings/NLP.jpg"\>
</p>

---

# Natural Language Processing — Word Embeddings

## Description

Ce projet aborde les techniques de représentation vectorielle du texte en traitement automatique du langage naturel (**NLP**). Il couvre les méthodes traditionnelles de comptage (**Bag of Words**, **TF-IDF**) jusqu'aux architectures d'apprentissage profond et d'embeddings contextuels (**Word2Vec**, **fastText**, **ELMo**).

L'objectif est d'implémenter ces algorithmes à la fois à partir de zéro avec **NumPy** et **scikit-learn**, et d'utiliser la bibliothèque **Gensim** et **TensorFlow/Keras** pour l'entraînement et l'extraction de vecteurs sémantiques.

---

## Technical Requirements

* **OS:** Ubuntu 20.04 LTS
* **Language:** Python 3.9
* **Main Libraries:**
* NumPy 1.25.2
* TensorFlow / Keras 2.15.0
* Gensim 4.3.3 (`pip install --user gensim==4.3.3`)
* scikit-learn


* **Style Guide:** Conformité stricte aux normes `pycodestyle` (v2.11.1)
* **Executable:** Tous les scripts exécutables commencent par `#!/usr/bin/env python3`
* **Documentation:** Tous les modules, classes et fonctions doivent être documentés.

---

## Key Concepts & Theory

* **Bag of Words (BoW) :** Représentation sous forme de matrice d'occurrence où chaque colonne correspond à un mot du vocabulaire et chaque ligne à un document/phrase.
* **TF-IDF (Term Frequency - Inverse Document Frequency) :** Mesure statistique évaluant l'importance d'un mot dans un document par rapport à un corpus complet.
* **Word2Vec (CBOW & Skip-Gram) :** Modèle de réseau de neurones à 2 couches apprenant des représentations vectorielles denses en prédisant un mot à partir de son contexte (**CBOW**) ou le contexte à partir d'un mot (**Skip-gram**).
* **Negative Sampling :** Technique d'optimisation pour Word2Vec permettant de mettre à jour seulement une petite fraction des poids au lieu de tout le vocabulaire à chaque étape.
* **fastText :** Extension de Word2Vec prenant en compte les n-grammes de sous-mots (*subwords*), permettant de générer des embeddings pour des mots hors vocabulaire (OOV).
* **ELMo (Embeddings from Language Models) :** Representations vectorielles profondes et **contextuelles** obtenues à l'aide d'un LSTM bidirectionnel pré-entraîné (BiLM).

---

## File Structure & Tasks Overview

| File | Description | Key Features |
| --- | --- | --- |
| `0-bag_of_words.py` | Crée une matrice d'embeddings Bag of Words à partir d'une liste de phrases (sans Gensim). | NumPy, regex tokenization, sorting |
| `1-tf_idf.py` | Calcule la matrice TF-IDF pour une liste de phrases donnée. | NumPy, math log, IDF formula |
| `2-word2vec.py` | Entraîne un modèle Word2Vec en utilisant Gensim. | `gensim.models.Word2Vec`, CBOW/Skip-gram |
| `3-extract_word2vec.py` | Extrait la matrice de poids ou le vecteur d'un mot spécifique d'un modèle Word2Vec. | Gensim `KeyedVectors` |
| `4-fasttext.py` | Entraîne un modèle fastText à l'aide de Gensim. | `gensim.models.FastText` |
| `5-elmo.py` | Génère des embeddings contextuels ELMo avec TensorFlow Hub / Keras. | TensorFlow 2.15, TF Hub |

---

## Installation & Execution

### 1. Préparation de l'environnement

```bash
cd holbertonschool-machine_learning/supervised_learning/word_embeddings
pip install --user gensim==4.3.3

```

### 2. Exemple d'exécution (Task 0 - Bag of Words)

```bash
chmod +x 0-main.py 0-bag_of_words.py
./0-main.py

```

### 3. Vérification de la conformité du code

```bash
pycodestyle *.py
python3 -c 'print(__import__("0-bag_of_words").bag_of_words.__doc__)'

```

---

## Author

* **Mathieu** — *Machine Learning Student @ Holberton School* — [GitHub Profile](https://github.com/Mathieu7483)