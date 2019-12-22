"""
"Convolutional Neural Networks for Sentence Classification"
https://arxiv.org/abs/1408.5882
"""

import json
import math
import os

from absl import app, flags, logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model  # pylint: disable=import-error
from tensorflow.keras.layers import Activation, Concatenate, Conv1D, Dense, Embedding, GlobalMaxPool1D, Input, LSTM  # pylint: disable=import-error
import tensorflow_datasets as tfds

from .preprocess import prepare_datasets


def prepare_model(embed_dim, l2_weight, vocab_size, conv_layers):
    seq = Input((None, ), dtype="int32")
    x = Embedding(
        vocab_size,
        embed_dim,
        embeddings_regularizer=tf.keras.regularizers.l2(l2_weight))(seq)

    conv_outputs = []
    for i, l in enumerate(conv_layers):
        c = Conv1D(
            filters=l[0],
            kernel_size=l[1],
            activation=None,
            kernel_regularizer=tf.keras.regularizers.l2(l2_weight),
        )(x)
        c = Activation("tanh", name="tanh_%d" % i)(c)
        c = GlobalMaxPool1D()(c)
        conv_outputs.append(c)

    x = Concatenate(axis=-1)(conv_outputs)
    x = Dense(
        2,
        activation=None,
        kernel_regularizer=tf.keras.regularizers.l2(l2_weight),
    )(x)  # from_logits

    return Model(seq, x)


def main(_):
    tf.random.set_seed(42)

    datasets = tfds.load("glue/sst2", data_dir=flags.FLAGS.tfds_data_dir)

    tokens = np.load("cd/tokens.npy")
    vocab_size = 2 + len(tokens)

    train_data, val_data = prepare_datasets(datasets,
                                            tokens=tokens,
                                            batch_size=flags.FLAGS.batch_size,
                                            val_min_len=8)

    conv_layers = json.loads(flags.FLAGS.conv_layers)
    model = prepare_model(embed_dim=flags.FLAGS.embed_dim,
                          l2_weight=flags.FLAGS.l2_weight,
                          vocab_size=vocab_size,
                          conv_layers=conv_layers)
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
    app.flags.DEFINE_integer("embed_dim", 128, "embed_dim")
    app.flags.DEFINE_float("l2_weight", 4 * 1e-4, "l2_weight")
    app.flags.DEFINE_integer("batch_size", 32, "batch_size")
    app.flags.DEFINE_string("conv_layers",
                            "[[128, 2], [128, 3], [128, 4], [128, 5]]",
                            "json encoded conv layers config")
    app.flags.DEFINE_integer("steps_per_epoch", int(math.ceil(67349 / 32)),
                             "steps_per_epoch")
    app.flags.DEFINE_string("tfds_data_dir", "~/tensorflow_datasets",
                            "tfds_data_dir")
    app.flags.DEFINE_string("job-dir", "./lstm", "job-dir")

    app.run(main)
