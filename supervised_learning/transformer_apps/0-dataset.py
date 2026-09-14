#!/usr/bin/env python3
"""Create the dataset class that loads and preps
the TED Talks dataset for training
a transformer model."""
import tensorflow_datasets as tfds
from setup import load_pt2en


class Dataset:
    """Loads and preps the TED Talks dataset for training
    a transformer model.
    Public instance attributes:
        data_train [tf.data.Dataset]: contains the ted_hrlr_pt_to_en
            training split, loaded as a tf.data.Dataset.
        data_valid [tf.data.Dataset]: contains the ted_hrlr_pt_to_en
            validation split, loaded as a tf.data.Dataset.
        tokenizer_pt [tfds.deprecated.text.SubwordTextEncoder]:
            Portuguese tokenizer created from the training set.
        tokenizer_en [tfds.deprecated.text.SubwordTextEncoder]:
            English tokenizer created from the training set.
    """
    def __init__(self):
        """Class constructor."""
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

    def tokenize_dataset(self, data):
        """Creates sub-word tokenizers for the dataset.
        Args:
            data [tf.data.Dataset]: contains the ted_hrlr_pt_to_en
                training split, loaded as a tf.data.Dataset.
        Returns: tokenizer_pt, tokenizer_en
            tokenizer_pt [tfds.deprecated.text.SubwordTextEncoder]:
                Portuguese tokenizer created from the training set.
            tokenizer_en [tfds.deprecated.text.SubwordTextEncoder]:
                English tokenizer created from the training set.
        """
        tokenizer_pt = (
            tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
                (pt.numpy() for pt, en in data), target_vocab_size=2**15
            )
        )
        tokenizer_en = (
            tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
                (en.numpy() for pt, en in data), target_vocab_size=2**15
            )
        )
        return tokenizer_pt, tokenizer_en
