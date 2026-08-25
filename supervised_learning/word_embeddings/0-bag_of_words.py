#!/usr/bin/env python3
"""Write a function that creates a bag of words embedding matrix"""
import re
import numpy as np


def bag_of_words(sentences, vocab=None):
    """Creates a bag of words embedding matrix"""
    cleaned_sentences = []
    for s in sentences:
        s_clean = s.lower()
        s_clean = s_clean.replace("'", "")
        for char in ".,?!\":;-":
            s_clean = s_clean.replace(char, ' ')
        words = s_clean.split()
        cleaned_sentences.append(words)

    if vocab is None:
        F = set()
        for words in cleaned_sentences:
            F.update(words)
        F = sorted(list(F))
    else:
        F = list(vocab)

    word_to_idx = {word: idx for idx, word in enumerate(F)}

    E = np.zeros((len(sentences), len(F)), dtype=int)

    for i, words in enumerate(cleaned_sentences):
        for word in words:
            if word in word_to_idx:
                E[i, word_to_idx[word]] += 1

    return E, F
