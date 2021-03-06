"""
TODO: word vectors initialized with uncased GloVe
"""

import math
import os

from absl import app, flags, logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model  # pylint: disable=import-error
from tensorflow.keras.layers import Dense, Dropout, Embedding, Input, LSTM  # pylint: disable=import-error
import tensorflow_datasets as tfds

from .preprocess import prepare_datasets


def prepare_model(embed_dim, hidden_dim, l2_weight, vocab_size):
    seq = Input((None, ), dtype="int32")
    x = Embedding(
        vocab_size,
        embed_dim,
        embeddings_regularizer=tf.keras.regularizers.l2(l2_weight),
    )(seq)
    x = LSTM(hidden_dim)(x)
    x = Dense(
        2,
        activation=None,
        kernel_regularizer=tf.keras.regularizers.l2(l2_weight),
    )(x)  # from_logits
    return tf.keras.Model(seq, x)


def main(_):
    tf.random.set_seed(42)

    datasets = tfds.load("glue/sst2", data_dir=flags.FLAGS.tfds_data_dir)

    tokens = np.load("cd/tokens.npy")
    vocab_size = 2 + len(tokens)

    train_data, val_data = prepare_datasets(datasets,
                                            tokens=tokens,
                                            batch_size=flags.FLAGS.batch_size)

    model = prepare_model(embed_dim=flags.FLAGS.embed_dim,
                          hidden_dim=flags.FLAGS.hidden_dim,
                          l2_weight=flags.FLAGS.l2_weight,
                          vocab_size=vocab_size)
    model.summary()

    optimizer = tf.keras.optimizers.Adam()

    job_dir = flags.FLAGS["job-dir"].value

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    acc = tf.keras.metrics.SparseCategoricalAccuracy()

    model.compile(optimizer, loss, metrics=[acc])

    train_steps = flags.FLAGS.steps_per_epoch
    val_steps = 872

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_sparse_categorical_accuracy",
            patience=10,
            restore_best_weights=True,
            verbose=1),
        tf.keras.callbacks.TensorBoard(log_dir=job_dir),
    ]

    model.fit(train_data,
              epochs=100,
              steps_per_epoch=train_steps,
              validation_data=val_data,
              validation_steps=val_steps,
              callbacks=callbacks)

    model.save(os.path.join(job_dir, "saved_model", "best"))


if __name__ == "__main__":
    app.flags.DEFINE_integer("embed_dim", 300, "embed_dim")
    app.flags.DEFINE_integer("hidden_dim", 168, "hidden_dim")
    app.flags.DEFINE_float("l2_weight", 4 * 1e-4, "l2_weight")
    app.flags.DEFINE_integer("batch_size", 32, "batch_size")
    app.flags.DEFINE_integer("steps_per_epoch", int(math.ceil(67349 / 32)),
                             "steps_per_epoch")
    app.flags.DEFINE_string("tfds_data_dir", "~/tensorflow_datasets",
                            "tfds_data_dir")
    app.flags.DEFINE_string("job-dir", "./lstm", "job-dir")

    app.run(main)
