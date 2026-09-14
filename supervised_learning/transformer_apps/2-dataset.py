#!/usr/bin/env python3
"""Create the dataset class that loads and preps
the TED Talks dataset for training
a transformer model."""
from setup import load_pt2en
import tensorflow as tf
import transformers


class Dataset:
    """Loads and preps the TED Talks dataset for training
    a transformer model.
    Public instance attributes:
        data_train [tf.data.Dataset]: contains the ted_hrlr_pt_to_en
            training split, loaded as a tf.data.Dataset.
        data_valid [tf.data.Dataset]: contains the ted_hrlr_pt_to_en
            validation split, loaded as a tf.data.Dataset.
        tokenizer_pt: Portuguese tokenizer created from the training set.
        tokenizer_en: English tokenizer created from the training set.
    """
    def __init__(self):
        """Class constructor."""
        self.data_train = load_pt2en('train')
        self.data_valid = load_pt2en('validation')

        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )

        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """Creates sub-word tokenizers for the dataset.
        Args:
            data [tf.data.Dataset]: contains the ted_hrlr_pt_to_en
                training split, loaded as a tf.data.Dataset.
        Returns: tokenizer_pt, tokenizer_en
            tokenizer_pt: Portuguese tokenizer created from the training set.
            tokenizer_en: English tokenizer created from the training set.
        """
        pt_corpus = (pt.numpy().decode('utf-8') for pt, en in data)
        en_corpus = (en.numpy().decode('utf-8') for pt, en in data)

        tokenizer_pt = transformers.AutoTokenizer.from_pretrained(
            'neuralmind/bert-base-portuguese-cased'
        )
        tokenizer_en = transformers.AutoTokenizer.from_pretrained(
            'bert-base-uncased'
        )

        tokenizer_pt = tokenizer_pt.train_new_from_iterator(
            pt_corpus, vocab_size=2**13
        )
        tokenizer_en = tokenizer_en.train_new_from_iterator(
            en_corpus, vocab_size=2**13
        )

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """Function that encodes a translation into tokens.
        pt : tf.Tensor containing the Portuguese sentence
        en : tf.Tensor containing the corresponding English sentence
        Returns: pt_tokens, en_tokens
        pt_tokens is a list containing the Portuguese tokens
        en_tokens is a list containing the English tokens
        """
        pt_sentence = pt.numpy().decode('utf-8')
        en_sentence = en.numpy().decode('utf-8')

        vocab_size_pt = self.tokenizer_pt.vocab_size
        vocab_size_en = self.tokenizer_en.vocab_size

        pt_tokens = self.tokenizer_pt.encode(
            pt_sentence, add_special_tokens=False
        )
        en_tokens = self.tokenizer_en.encode(
            en_sentence, add_special_tokens=False
        )

        pt_tokens = [vocab_size_pt] + pt_tokens + [vocab_size_pt + 1]
        en_tokens = [vocab_size_en] + en_tokens + [vocab_size_en + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """Acts as a tensorflow wrapper for the encode instance method."""
        pt_tokens, en_tokens = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )

        pt_tokens.set_shape([None])
        en_tokens.set_shape([None])

        return pt_tokens, en_tokens