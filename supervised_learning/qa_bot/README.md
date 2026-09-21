<p align="center"\>
<img src="https://github.com/Mathieu7483/holbertonschool-machine_learning/blob/main/supervised_learning/qa_bot/QA%20Bots.png"\>
</p>

# QA Bot — Question Answering & Semantic Search

## Description

Ce projet porte sur la conception et le développement d'un chatbot de réponse aux questions (**QA Bot**) capable d'extraire des réponses précises au sein d'une base de connaissances documentaire (articles Markdown issus de la documentation Zendesk d'Holberton).

Le système s'appuie sur une combinaison de modèles de **Deep Learning** et de techniques de **Natural Language Processing (NLP)** moderne :

* **Question-Answering Extractif :** Extraction directe des segments de texte pertinents à partir d'un document de référence à l'aide de modèles pré-entraînés BERT (*SQuAD* fine-tuned).
* **Recherche Sémantique (*Semantic Search*) :** Vectorisation des documents et des requêtes utilisateur via des embeddings pour identifier les articles les plus pertinents au sein d'un corpus étendu.
* **Pipeline QA Multi-Référence :** Traitement de bout en bout permettant au bot de sélectionner le bon document source puis d'en extraire la réponse appropriée lors d'une interaction en boucle fermée.

---

## Technical Requirements

* **OS:** Ubuntu 20.04 LTS
* **Language:** Python 3.9
* **Main Libraries:**
* NumPy 1.25.2
* TensorFlow 2.15
* TensorFlow Hub 0.15.0 (`tensorflow-hub`)
* Transformers 4.44.2 (`transformers`)


* **Style Guide:** Conformité stricte aux normes `pycodestyle` (v2.11.1)
* **Executable:** Tous les fichiers exécutables commencent par `#!/usr/bin/env python3`
* **Documentation:** Tous les modules, classes et fonctions sont intégralement documentés.

---

## Environment Setup & Dataset

### 1. Installation des dépendances

```bash
pip install --user tensorflow-hub==0.15.0 transformers==4.44.2

```

### 2. Preparation de la base de connaissances

Extraire l'archive `ZendeskArticles.zip` contenant les articles de référence au format Markdown à la racine du dossier du projet.

```bash
unzip ZendeskArticles.zip -d ZendeskArticles/

```

---

## Key Concepts & Architecture

* **Extractive Question Answering :** Utilisation du modèle `bert-uncased-tf2-qa` (TensorFlow Hub) et du tokenizer `bert-large-uncased-whole-word-masking-finetuned-squad` (Hugging Face) pour localiser les indices de début (*start logits*) et de fin (*end logits*) de la réponse dans la séquence.
* **Semantic Search & Sentence Embeddings :** Conversion du texte en vecteurs de représentation dense pour calculer la similarité cosinus (*Cosine Similarity*) entre la question posée et les documents de la base de connaissances.
* **Interactive CLI Loop :** Interface utilisateur en ligne de commande gérant la saisie utilisateur, les mots-clés de sortie (`exit`, `quit`, `bye`) et l'affichage fluide des réponses.

---

## File Structure & Tasks Overview

| File | Description | Key Components / Concepts |
| --- | --- | --- |
| `0-qa.py` | Extraction de réponse au sein d'un document unique. | `bert-uncased-tf2-qa`, `BertTokenizerFast`, Start/End Logits |
| `1-loop.py` | Gestion de la boucle d'interaction CLI avec l'utilisateur. | Read-Eval-Print Loop (REPL), Exit conditions handling |
| `2-qa.py` | Intégration de la recherche de réponse dans la boucle interactive. | Combining `0-qa.py` functionality with interactive CLI |
| `3-semantic_search.py` | Recherche du document le plus pertinent dans un dossier via embeddings. | Document Vectorization, Cosine Similarity scoring |
| `4-qa.py` | QA Bot complet multi-références (Sélection du document + Extraction de réponse). | Full Pipeline: Semantic Search $\rightarrow$ QA Extraction $\rightarrow$ CLI Output |

---

## Usage Example (Task 0 - Question Answering)

```bash
chmod +x 0-main.py 0-qa.py
./0-main.py

```

### Script Example (`0-main.py`)

```python
#!/usr/bin/env python3
question_answer = __import__('0-qa').question_answer

with open('ZendeskArticles/PeerLearningDays.md') as f:
    reference = f.read()

print(question_answer('When are PLDs?', reference))
# Output: on - site days from 9 : 00 am to 3 : 00 pm

```

---

## ✍️ Author

  * **Mathieu** - *Programming student, specialization Machine Learning* - [👤 My Github profile](https://github.com/Mathieu7483)
