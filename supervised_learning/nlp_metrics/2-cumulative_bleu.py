#!/usr/bin/env python3
"""Calculates the cumulative BLEU score for a sentence"""
import numpy as np


def cumulative_bleu(references, sentence, n):
    """Calculates the cumulative n-gram BLEU score for a sentence

    Args:
        references (list): list of reference translations, where each reference
                           is a list of words.
        sentence (list): list containing the model proposed sentence.
        n (int): size of the largest n-gram to use for evaluation

    Returns:
        float: the cumulative n-gram BLEU score
    """
    c = len(sentence)
    if c == 0:
        return 0.0

    # 1. Calculate the effective reference length r
    precisions = []

    # Loop through each n-gram order from 1 to n
    for i in range(1, n + 1):
        if c < i:
            precisions.append(0.0)
            continue

        # Extraction of i-grams from the candidate sentence
        sentence_ngrams = {}
        for j in range(c - i + 1):
            ngram = tuple(sentence[j:j + i])
            sentence_ngrams[ngram] = sentence_ngrams.get(ngram, 0) + 1

        # Extraction of i-grams from references
        ref_ngrams_list = []
        for ref in references:
            ref_dict = {}
            for j in range(len(ref) - i + 1):
                ref_ngram = tuple(ref[j:j + i])
                ref_dict[ref_ngram] = ref_dict.get(ref_ngram, 0) + 1
            ref_ngrams_list.append(ref_dict)

        # Calculate of Clipped Count
        clipped_count = 0
        for ngram, count in sentence_ngrams.items():
            max_ref_count = max([ref_dict.get(ngram, 0)
                                for ref_dict in ref_ngrams_list])
            clipped_count += min(count, max_ref_count)

        precision = clipped_count / (c - i + 1)
        precisions.append(precision)

    # 2. Geometric Mean of the modified precisions
    if any(p == 0 for p in precisions):
        geo_mean = 0.0
    else:
        weights = np.full(n, 1 / n)
        geo_mean = np.exp(np.sum(weights * np.log(precisions)))

    # 3. Brevity Penalty
    ref_lens = [len(ref) for ref in references]
    r = min(ref_lens, key=lambda ref_len: (abs(ref_len - c), ref_len))

    if c > r:
        bp = 1.0
    else:
        bp = np.exp(1 - r / c)

    return bp * geo_mean
