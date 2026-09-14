#!/usr/bin/env python3
"""Create the dataset class that loads and preps
the TED Talks dataset for training
a transformer model."""
from setup import load_pt2en
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
