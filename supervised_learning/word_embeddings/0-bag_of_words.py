#!/usr/bin/env python3
"""Write a function that creates a bag of words embedding matrix"""
import re
import numpy as np


def bag_of_words(sentences, vocab=None):
    """Creates a bag of words embedding matrix"""
    cleaned_sentences = []
    for sentence in sentences:
        words = re.findall(r'\b\w+\b', sentence.lower())
        cleaned_sentences.append(words)

    if vocab is None:
        vocab_set = set()
        for words in cleaned_sentences:
            vocab_set.update(words)
        F = sorted(list(vocab_set))
    else:
        F = list(vocab)

    word_to_idx = {word: idx for idx, word in enumerate(F)}

    E = np.zeros((len(sentences), len(F)), dtype=int)

    for i, words in enumerate(cleaned_sentences):
        for word in words:
            if word in word_to_idx:
                E[i, word_to_idx[word]] += 1

    return E, F
