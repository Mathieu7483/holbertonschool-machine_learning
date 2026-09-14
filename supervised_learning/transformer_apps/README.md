<p align="center"\>
<img src="https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/supervised_learning/transformer_apps/Transformer%20applications.jpg"\>
</p>

# Transformer Applications — Dataset & Pipeline Prep

## Description

Ce projet porte sur l'application pratique de l'architecture **Transformer** dans le cadre de la traduction automatique (**Machine Translation**) du portugais vers l'anglais.

Il couvre l'intégralité du traitement des données NLP modernes sous **TensorFlow / Keras** et la librairie **Hugging Face Transformers** :

* Manipulation et chargement du dataset parallèle *TED Talks* (`ted_hrlr_translate/pt_to_en`).
* Entraînement de tokenizers sous-mots (*Sub-word Tokenization*) à partir de modèles BERT pré-entraînés.
* Encodage dynamique des séquences textuelles au sein de pipelines `tf.data.Dataset`.
* Génération des masques d'attention (*Padding Mask* et *Look-ahead / Causal Mask*).
* Construction d'une boucle d'entraînement sur-mesure pour un réseau Transformer complet.

---

## Technical Requirements

* **OS:** Ubuntu 20.04 LTS
* **Language:** Python 3.9
* **Main Libraries:**
* NumPy 1.25.2
* TensorFlow 2.15
* Transformers 4.44.2 (`transformers`)
* TensorFlow Datasets 4.9.2 (`tensorflow-datasets`)


* **Style Guide:** Conformité stricte à la norme `pycodestyle` (v2.11.1)
* **Executable:** Tous les scripts exécutables commencent par `#!/usr/bin/env python3`
* **Documentation:** Modules, classes et fonctions entièrement documentés.

---

## Technical Setup & Mirror Helper

L'archive d'origine de `ted_hrlr_translate` étant indisponible sur `phontron.com`, nous utilisons un miroir auto-hébergé avec un helper `setup.py`.

### 1. Téléchargement de l'archive et de setup.py

```bash
# Téléchargement et extraction du dataset dans le cache
curl -L -O https://holbucket-prod.s3.fr-par.scw.cloud/projects/2422/ted_hrlr_pt_to_en.tar.gz
mkdir -p ~/.cache/ted_hrlr
tar -xzvf ted_hrlr_pt_to_en.tar.gz -C ~/.cache/ted_hrlr

# Téléchargement du helper setup.py à la racine du projet
curl -L -O https://holbucket-prod.s3.fr-par.scw.cloud/projects/2422/setup.py

```

### 2. Dépendances requises

```bash
pip install --user tensorflow-datasets==4.9.2 transformers==4.44.2

```

---

## Key Concepts & Architecture

* **Subword Tokenization (`BertTokenizerFast`) :** Utilisation d'un algorithme de découpage en sous-mots (WordPiece) entraîné sur le corpus cible à partir d'un vocabulaire de départ de taille maximale $2^{13} = 8192$.
* **`tf.py_function` Integration :** Encapsulation de fonctions Python/HuggingFace dans le graphe d'exécution TensorFlow pour permettre le traitement dynamique des batchs.
* **`tf.data.Dataset` Pipeline :** Optimisation du chargement avec `map()`, `filter()`, `cache()`, `shuffle()`, `padded_batch()` et `prefetch()`.
* **Attention Masks :**
* **Padding Mask :** Masque les jetons de rembourrage (`0`) pour éviter qu'ils n'impactent le calcul de l'attention.
* **Look-ahead Mask :** Masque triangulaire supérieur empêchant le décodeur d'accéder aux jetons futurs lors du mode causal.



---

## File Structure & Tasks Overview

| File | Description | Key Components / Concepts |
| --- | --- | --- |
| `0-dataset.py` | Classe `Dataset` pour le chargement et l'entraînement des tokenizers sub-word. | `load_pt2en`, `AutoTokenizer.train_new_from_iterator` |
| `1-dataset.py` | Implémentation de la méthode d'encodage des phrases en tokens (`encode`). | Add `[CLS]` / `[SEP]` tokens, Subword token IDs |
| `2-dataset.py` | Encapsulation TensorFlow de la méthode d'encodage (`tf_encode`). | `tf.py_function`, Tensor shape & type preservation |
| `3-dataset.py` | Pipeline `tf.data.Dataset` complet pré-traité et prêt pour l'entraînement. | `map`, `filter`, `padded_batch`, `prefetch` |
| `4-create_masks.py` | Fonction de génération de tous les masques requis par le Transformer. | Padding Mask, Look-ahead Mask (`tf.linalg.band_part`) |
| `5-train.py` | Boucle d'entraînement personnalisée (*Custom Training Loop*) du modèle Transformer. | Custom Keras `fit` / `train_step`, Cross-Entropy Loss, Accuracy |

---

## Execution Example (Task 0 - Dataset Initialization)

```bash
chmod +x 0-main.py 0-dataset.py
./0-main.py

```

---

## ✍️ Author

  * **Mathieu** - *Programming student, specialization Machine Learning* - [👤 My Github profile](https://github.com/Mathieu7483)