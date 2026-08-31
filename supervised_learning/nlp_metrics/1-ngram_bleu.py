#!/usr/bin/env python3
"""Calculates the n-gram BLEU score for a sentence"""
import numpy as np


def ngram_bleu(references, sentence, n):
    """Calculates the n-gram BLEU score for a sentence

    Args:
        references (list): list of reference translations, where each reference
                           is a list of words.
        sentence (list): list containing the model proposed sentence.
        n (int): size of the n-gram to use for evaluation

    Returns:
        float: the n-gram BLEU score
    """
    c = len(sentence)
    # condition to avoid division by zero when calculating precision
    if c < n:
        return 0.0

    # 1. Calculate the effective reference length r
    ref_lens = [len(ref) for ref in references]
    r = min(ref_lens, key=lambda ref_len: (abs(ref_len - c), ref_len))

    # 2. Extraction and counting n-grams from the candidate sentence
    sentence_ngrams = {}
    for i in range(c - n + 1):
        ngram = tuple(sentence[i:i + n])
        sentence_ngrams[ngram] = sentence_ngrams.get(ngram, 0) + 1

    # 3. Extract and count n-grams from references
    ref_ngrams_list = []
    for ref in references:
        ref_dict = {}
        for i in range(len(ref) - n + 1):
            ref_ngram = tuple(ref[i:i + n])
            ref_dict[ref_ngram] = ref_dict.get(ref_ngram, 0) + 1
        ref_ngrams_list.append(ref_dict)

    # 4. Calculate the Clipped Count
    clipped_count = 0
    for ngram, count in sentence_ngrams.items():
        max_ref_count = max([ref_dict.get(ngram, 0)
                             for ref_dict in ref_ngrams_list])
        clipped_count += min(count, max_ref_count)

    total_candidate_ngrams = c - n + 1
    precision = clipped_count / total_candidate_ngrams

    # 5. Brevity Penalty (BP)
    if c > r:
        bp = 1.0
    else:
        bp = np.exp(1 - r / c)

    return bp * precision
