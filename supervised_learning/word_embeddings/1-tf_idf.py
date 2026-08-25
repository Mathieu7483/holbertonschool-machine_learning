#!/usr/bin/env python3
"""creates a TF-IDF embedding matrix"""
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np


def tf_idf(sentences, vocab=None):
    """creates a TF-IDF embedding matrix from a list of sentences"""
    if vocab is not None:
        vectorizer = TfidfVectorizer(vocabulary=vocab)
    else:
        vectorizer = TfidfVectorizer()

    X = vectorizer.fit_transform(sentences)
    E = X.toarray()
    
    if vocab is None:
        F = np.array(vectorizer.get_feature_names_out())
    else:
        F = np.array(vocab)

    return E, F
