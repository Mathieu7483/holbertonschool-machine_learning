#!/usr/bin/env python3
"""
Module defining semantic_search function using SentenceTransformer.
"""
import os
from sentence_transformers import SentenceTransformer, util


def semantic_search(corpus_path, sentence):
    """
    Performs semantic search on a corpus of documents to find
    the text most similar to a given sentence using SentenceTransformers.

    Args:
        corpus_path (str): Path to the corpus directory.
        sentence (str): Query sentence.

    Returns:
        str: Reference text of the most similar document.
    """
    # Chargement du modèle SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')

    documents = []

    # Tri pour assurer la consistance des index
    filenames = sorted(os.listdir(corpus_path))

    for filename in filenames:
        if filename.startswith('.'):
            continue
        file_path = os.path.join(corpus_path, filename)

        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                documents.append(f.read())

    if not documents:
        return None

    # Calcul des embeddings pour la question et tous les documents
    query_embedding = model.encode(sentence, convert_to_tensor=True)
    doc_embeddings = model.encode(documents, convert_to_tensor=True)

    # Calcul de la similarité cosinus via l'utilitaire dédié
    cosine_scores = util.cos_sim(query_embedding, doc_embeddings)[0]

    # Récupération de l'index du meilleur document
    best_idx = cosine_scores.argmax().item()

    return documents[best_idx]
