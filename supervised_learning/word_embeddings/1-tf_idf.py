#!/usr/bin/env python3
"""write a function that creates a TF-IDF embedding matrix"""
import re
import string
import numpy as np


def tf_idf(sentences, vocab=None):
    """creates a TF-IDF embedding matrix from a list of sentences"""
    # Preprocessing the sentences
    processed_sentences = []
    for sentence in sentences:
        s = sentence.lower()
        s = re.sub(r"\'s\b", "", s)
        s = re.sub(f"[{re.escape(string.punctuation)}]", "", s)
        processed_sentences.append(s.split())

    # Build vocab if None
    if vocab is None:
        vocab = sorted(set(
            word for s in processed_sentences for word in s
        ))

    # Initialize word_to_idx map and matrices
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    tf = np.zeros((len(sentences), len(vocab)), dtype=float)
    num_docs = len(sentences)

    for i, s in enumerate(processed_sentences):
        total_words = len(s)
        if total_words > 0:
            for word in s:
                if word in word_to_idx:
                    tf[i, word_to_idx[word]] += 1
            # Term Frequency (TF)
            tf[i] /= total_words

    # Document Frequency (DF)
    df = np.sum(tf > 0, axis=0)

    # Inverse Document Frequency (IDF)
    with np.errstate(divide='ignore'):
        idf = np.log(num_docs / df)
        idf[np.isinf(idf) | np.isnan(idf)] = 0.0

    # TF-IDF Embedding
    E = tf * idf

    return E, np.array(vocab)
