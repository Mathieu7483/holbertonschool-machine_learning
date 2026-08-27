<p align="center"\>
<img src="https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/supervised_learning/nlp_metrics/Metrics%20evaluation%20NLP.jpg"\>
</p>

---

# Natural Language Processing — Evaluation Metrics

## Description

Ce projet porte sur l'implémentation algorithmique *from scratch* des métriques clés d'évaluation des modèles de traitement automatique du langage naturel (**NLP**). L'objectif est de comprendre en profondeur le fonctionnement mathématique et pratique de métriques telles que **BLEU**, **ROUGE** et la **Perplexité**, souvent utilisées pour évaluer la traduction automatique, le résumé de texte et la génération de langage.

Toutes les fonctions sont codées en **Python 3** avec **NumPy**, sans dépendre de modules externes tels que `nltk`.

---

## Technical Requirements

* **OS:** Ubuntu 20.04 LTS
* **Language:** Python 3.9
* **Main Library:** NumPy 1.25.2 (Le module `nltk` est strictement interdit)
* **Style Guide:** Conformité stricte aux normes `pycodestyle` (v2.11.1)
* **Executable:** Tous les scripts exécutables commencent par `#!/usr/bin/env python3`
* **Documentation:** Modules, classes et fonctions entièrement documentés.

---

## Key Concepts & Theory

* **BLEU (Bilingual Evaluation Understudy) :** Mesure la précision des $n$-grammes d'une traduction proposée par rapport à une ou plusieurs traductions de référence, enrichie d'une pénalité de brièveté (*Brevity Penalty*).
* **ROUGE (Recall-Oriented Understudy for Gisting Evaluation) :** Métrique principalement axée sur le rappel, mesurant le chevauchement de $n$-grammes ou la plus longue sous-séquence commune (LCS) pour le résumé automatique.
* **Perplexity :** Évalue la qualité d'un modèle de langue en mesurant la certitude avec laquelle il prédit un échantillon de texte (plus elle est basse, meilleur est le modèle).
* **Brevity Penalty (BP) :** Facteur appliqué au score BLEU pour éviter d'avantager les phrases générées trop courtes qui obtiendraient artificiellement une haute précision.

---

## File Structure & Tasks Overview

| File | Description | Key Formulas / Concepts |
| --- | --- | --- |
| `0-uni_bleu.py` | Calcule le score BLEU 1-gramme (Unigram) pour une phrase par rapport à des références. | Precision $p_1$, Brevity Penalty $BP$ |
| `1-ngram_bleu.py` | Calcule le score BLEU pour un $n$-gramme spécifique ($n > 1$). | Modified $n$-gram precision $p_n$ |
| `2-cumulative_bleu.py` | Calcule le score BLEU cumulé de $1$ à $N$-grammes avec des poids définis. | $BP \cdot \exp\left(\sum w_n \log p_n\right)$ |

---

## Installation & Usage

### 1. Préparation du répertoire

```bash
cd holbertonschool-machine_learning/supervised_learning/nlp_metrics

```

### 2. Exemple d'exécution (Task 0 - Unigram BLEU)

```bash
chmod +x 0-main.py 0-uni_bleu.py
./0-main.py

```

### 3. Vérification de la conformité du code

```bash
pycodestyle *.py
python3 -c 'print(__import__("0-uni_bleu").uni_bleu.__doc__)'

```

---

## Author

* **Mathieu** — *Machine Learning Student @ Holberton School* — [GitHub Profile](https://github.com/Mathieu7483)