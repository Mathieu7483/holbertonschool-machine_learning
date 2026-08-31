#!/usr/bin/env python3
"""Write the function that calculates the unigram BLEU score for a sentence"""
import numpy as np


def uni_bleu(references, sentence):
    """Calculates the unigram BLEU score for a sentence

    Args:
        references (list): is a list of reference translations
        each reference translation is a list of the words
        in the translation
        sentence (list): list containing the model proposed sentence

    Returns:
        float: the unigram BLEU score
    """
    # Count the number of words in the sentence
    sentence_count = len(sentence)
    # Count the number of words in the references
    reference_count = sum(len(ref) for ref in references)
    # Create a dictionary to store the counts of each word in the references
    reference_word_counts = {}
    for ref in references:
        for word in ref:
            if word in reference_word_counts:
                reference_word_counts[word] += 1
            else:
                reference_word_counts[word] = 1
    # Create a dictionary to store the counts of each word in the sentence
    sentence_word_counts = {}
    for word in sentence:
        if word in sentence_word_counts:
            sentence_word_counts[word] += 1
        else:
            sentence_word_counts[word] = 1
    # Calculate the number of words in the sentence that are also in the references
    matching_word_count = 0
    for word in sentence_word_counts:
        if word in reference_word_counts:
            matching_word_count += min(sentence_word_counts[word], reference_word_counts[word])
    # Calculate the unigram precision
    precision = matching_word_count / sentence_count if sentence_count > 0 else 0
    # Calculate the brevity penalty
    brevity_penalty = 1 if sentence_count > reference_count else np.exp(1 - reference_count / sentence_count) if sentence_count > 0 else 0
    # Calculate the unigram BLEU score
    bleu_score = brevity_penalty * precision
    return bleu_score
