#!/usr/bin/env python3
"""
Creates a word2vec embedding matrix
"""
import gensim


def word2vec_model(sentences, vector_size=100, min_count=5,
                   window=5, negative=5, cbow=True, epochs=5,
                   seed=0, workers=1):
    """creates a word2vec embedding matrix from a list of sentences"""
    sg = 0 if cbow else 1
    model = gensim.models.Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        negative=negative,
        sg=sg,
        epochs=epochs,
        seed=seed,
        workers=workers
    )
    return model
