<p align="center"\>
<img src="https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/supervised_learning/attention/Attention.png"\>
</p>

# Natural Language Processing — Attention Mechanisms & Transformers

## Description

Ce projet explore l'évolution des architectures de traitement du langage naturel (**NLP**), des mécanismes d'attention appliqués aux réseaux récurrents (**Seq2Seq mit Bahdanau/Luong Attention**) jusqu'à la construction complète *from scratch* d'un **Transformer** sous **TensorFlow / Keras**.

L'objectif est d'implémenter brique par brique chaque composant des architectures modernes : encodage positionnel (*Positional Encoding*), attention sous toutes ses formes (*Scaled Dot-Product*, *Multi-Head Attention*), blocs d'encodage/dÉcodage et l'assemblage complet d'un réseau Transformer.

---

## Technical Requirements

* **OS:** Ubuntu 20.04 LTS
* **Language:** Python 3.9
* **Main Libraries:**
* NumPy 1.25.2
* TensorFlow 2.15 (utilisation exclusive de `tf.keras`)


* **Style Guide:** Conformité stricte aux normes `pycodestyle` (v2.11.1)
* **Executable:** Tous les scripts exécutables commencent par `#!/usr/bin/env python3`
* **Documentation:** Tous les modules, classes et fonctions sont documentés.

---

## Key Concepts & Theory

* **Attention Mechanism :** Permet au modèle d'attribuer des poids dynamiques à différentes parties de la séquence d'entrée en fonction du contexte courant.
* **Scaled Dot-Product Attention :** Calcul du score d'attention via le produit matriciel des Query ($Q$) et Key ($K$), divisé par $\sqrt{d_k}$, appliqué aux Values ($V$).
* **Multi-Head Attention :** Division de $Q, K, V$ en plusieurs sous-espaces de projection (*heads*) pour capturer des relations d'attention simultanées et variées.
* **Positional Encoding :** Injecte l'information de position relative/absolue des tokens dans la séquence à l'aide de fonctions sinusoïdales (sin/cos).
* **Self-Supervised Learning :** Pré-entraînement sur de grands volumes de texte sans étiquetage manuel (ex: Masked Language Modeling avec BERT, Causal Language Modeling avec GPT).

---

## File Structure & Tasks Overview

| File | Description | Key Components / Concepts |
| --- | --- | --- |
| `0-rnn_encoder.py` | Encodeur récurrent à base de GRU pour la traduction automatique. | `tf.keras.layers.Embedding`, `tf.keras.layers.GRU` |
| `1-self_attention.py` | Calcul de l'attention de Bahdanau (Self Attention pour RNN). | Dense layers ($W, U, V$), Softmax weights |
| `2-rnn_decoder.py` | Décodeur récurrent intégrant l'attention de Bahdanau. | Context vector, GRU, Softmax output |
| `3-positional_encoding.py` | Génération de la matrice d'encodage positionnel sinusoïdal. | Sine/Cosine formulas on positional indices |
| `4-scaled_dot_product_attention.py` | Implémentation du Scaled Dot-Product Attention avec masquage. | $Q, K, V$, Masking, Softmax scaling |
| `5-mha.py` | Layer de Multi-Head Attention complète. | Linear projections, Head splitting, Concatenation |
| `6-transformer_encoder_block.py` | Bloc unitaire d'encodeur Transformer. | Multi-Head Attention, Feed Forward, LayerNorm, Dropout |
| `7-transformer_decoder_block.py` | Bloc unitaire de décodeur Transformer. | Masked Multi-Head Attention, Cross Attention |
| `8-transformer_encoder.py` | Encodeur Transformer complet (*N* blocs + Positional Encoding). | Stacked Encoder Blocks, Embedding |
| `9-transformer_decoder.py` | Décodeur Transformer complet (*N* blocs + Positional Encoding). | Stacked Decoder Blocks, Embedding |
| `10-transformer_network.py` | Réseau Transformer complet (Encoder + Decoder + Linear classifier). | Full Transformer Architecture |

---

## Installation & Usage

### 1. Préparation du répertoire

```bash
cd holbertonschool-machine_learning/supervised_learning/attention

```

### 2. Exemple d'exécution (Task 0 - RNN Encoder)

```bash
chmod +x 0-main.py 0-rnn_encoder.py
./0-main.py

```

### 3. Vérification de la conformité du code

```bash
pycodestyle *.py
python3 -c 'print(__import__("0-rnn_encoder").RNNEncoder.__doc__)'
python3 -c 'print(__import__("0-rnn_encoder").RNNEncoder.call.__doc__)'

```

---

## Author

* **Mathieu** — *Machine Learning Student @ Holberton School* — [GitHub Profile](https://github.com/Mathieu7483)