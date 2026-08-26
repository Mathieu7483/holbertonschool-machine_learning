#!usr/bin/env python3
"""creates a fasttext embedding matrix"""
import gensim


def fasttext_model(sentences, vector_size=100, min_count=5,
                   window=5, negative=5, cbow=True, epochs=5,
                   seed=0, workers=1):
    """creates a fasttext embedding matrix from a list of sentences"""
    # Choose training algorithm: CBOW (sg=0) or Skip-gram (sg=1)
    sg = 0 if cbow else 1

    # Initialize the FastText model with the specified parameters
    model = gensim.models.FastText(
        sentences=sentences,
        vector_size=vector_size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg,
        epochs=epochs,
        seed=seed,
        workers=workers
    )

    # Build vocabulary from the sentences and train the model
    model.build_vocab(sentences)
    model.train(sentences, total_examples=model.corpus_count,
                epochs=model.epochs)

    return model
