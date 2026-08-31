#!/usr/bin/env python3
"""Write the function that calculates the unigram BLEU score for a sentence"""
import numpy as np


def uni_bleu(references, sentence):
    """Calculates the unigram BLEU score for a sentence

    Args:
        references (list): list of reference translations, where each reference
                           is a list of words.
        sentence (list): list containing the model proposed sentence.

    Returns:
        float: the unigram BLEU score
    """
    c = len(sentence)
    if c == 0:
        return 0.0

    # 1. Calculate the effective reference length r
    ref_lens = [len(ref) for ref in references]
    # find the reference length that is closest to the candidate length c
    r = min(ref_lens, key=lambda ref_len: (abs(ref_len - c), ref_len))

    # 2. Clipped Precision for unigrammes
    sentence_word_counts = {}
    for word in sentence:
        sentence_word_counts[word] = sentence_word_counts.get(word, 0) + 1

    clipped_count = 0
    for word, count in sentence_word_counts.items():
        # find the maximum count of the word in any reference
        max_ref_count = max([ref.count(word) for ref in references])
        clipped_count += min(count, max_ref_count)

    precision = clipped_count / c

    # 3. Brevity Penalty (BP)
    if c > r:
        bp = 1.0
    else:
        bp = np.exp(1 - r / c)

    return bp * precision
