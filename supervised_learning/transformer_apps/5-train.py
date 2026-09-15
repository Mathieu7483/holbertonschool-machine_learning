#!/usr/bin/env python3
"""Train a Transformer model for Portuguese to English translation."""
import tensorflow as tf
Dataset = __import__('3-dataset').Dataset
create_masks = __import__('4-create_masks').create_masks
Transformer = __import__('5-transformer').Transformer


class LearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Custom Learning Rate Scheduler for Transformer."""
    def __init__(self, dm, warmup_steps=4000):
        super(LearningRateSchedule, self).__init__()
        self.dm = tf.cast(dm, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.dm) * tf.math.minimum(arg1, arg2)


def loss_function(real, pred, loss_object):
    """Calculates Sparse Categorical Crossentropy loss ignoring padding."""
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def accuracy_function(real, pred):
    """Calculates accuracy ignoring padding tokens."""
    accuracies = tf.equal(real, tf.cast(tf.argmax(pred, axis=-1), tf.int64))
    mask = tf.math.logical_not(tf.math.equal(real, 0))

    accuracies = tf.math.logical_and(mask, accuracies)
    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)

    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):
    """Creates and trains a transformer model."""
    data = Dataset(batch_size, max_len)

    input_vocab_size = data.tokenizer_pt.vocab_size + 2
    target_vocab_size = data.tokenizer_en.vocab_size + 2

    transformer = Transformer(
        N, dm, h, hidden,
        input_vocab_size, target_vocab_size,
        max_len, max_len
    )

    learning_rate = LearningRateSchedule(dm)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate,
        beta_1=0.9,
        beta_2=0.98,
        epsilon=1e-9
    )

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction='none'
    )

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    for epoch in range(epochs):
        train_loss.reset_states()
        train_accuracy.reset_states()

        for (batch, (inp, tar)) in enumerate(data.data_train):
            tar_inp = tar[:, :-1]
            tar_real = tar[:, 1:]

            enc_mask, combined_mask, dec_mask = create_masks(inp, tar_inp)

            with tf.GradientTape() as tape:
                predictions = transformer(
                    inp, tar_inp, True,
                    enc_mask, combined_mask, dec_mask
                )
                loss = loss_function(tar_real, predictions, loss_object)

            gradients = tape.gradient(loss, transformer.trainable_variables)
            optimizer.apply_gradients(
                zip(gradients, transformer.trainable_variables)
            )

            acc = accuracy_function(tar_real, predictions)
            train_loss(loss)
            train_accuracy(acc)

            if batch % 50 == 0:
                print(f"Epoch {epoch + 1}, Batch {batch}: "
                      f"Loss {train_loss.result()}, "
                      f"Accuracy {train_accuracy.result()}")

        print(f"Epoch {epoch + 1}: "
              f"Loss {train_loss.result()}, "
              f"Accuracy {train_accuracy.result()}")

    return transformer
