import math
import os

from absl import app, flags, logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model  # pylint: disable=import-error
from tensorflow.keras.layers import Dense, Embedding, Input, LSTM  # pylint: disable=import-error
import tensorflow_datasets as tfds


def prepare_model(vocab_size):
    seq = Input((None, ), dtype="int32")
    x = Embedding(vocab_size, 256)(seq)
    x = LSTM(128)(x)
    x = Dense(2, activation=None)(x)
    return tf.keras.Model(seq, x)


def create_table(tokens):
    keys = tokens
    values = 1 + tf.range(len(keys))
    return tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, values), 1 + len(keys))


def preprocess_dataset(dataset, table):
    dataset = dataset.map(lambda x: (x["sentence"], x["label"]))
    dataset = dataset.map(lambda a, b: (tf.map_fn(
        lambda x: table.lookup(x), tf.strings.split(a), dtype=tf.int32), b))
    return dataset


def main(_):
    tf.random.set_seed(42)

    datasets = tfds.load("glue/sst2", data_dir=flags.FLAGS.tfds_data_dir)

    train_data = datasets[tfds.Split.TRAIN]
    val_data = datasets[tfds.Split.VALIDATION]

    tokens = np.load("cd/tokens.npy")

    table = create_table(tokens)

    train_data = preprocess_dataset(train_data, table)
    train_data = train_data.cache()
    train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)
    train_data = train_data.shuffle(buffer_size=1000)
    train_data = train_data.padded_batch(32, padded_shapes=((None, ), ()))
    train_data = train_data.repeat()

    val_data = preprocess_dataset(val_data, table)
    val_data = val_data.padded_batch(32, padded_shapes=((None, ), ()))
    val_data = val_data.repeat()

    model = prepare_model(2 + len(tokens))
    optimizer = tf.keras.optimizers.Adam()

    job_dir = flags.FLAGS["job-dir"].value
    checkpoint_dir = os.path.join(job_dir, "checkpoints")
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    _ = checkpoint.restore(latest_checkpoint)
    initial_epoch = 0
    if latest_checkpoint != None:
        logging.info("Restoring from: %s", latest_checkpoint)
        initial_epoch = int(latest_checkpoint.split("-")[-1])
    else:
        logging.info("Starting from Scratch")

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    acc = tf.keras.metrics.SparseCategoricalAccuracy()

    model.compile(optimizer, loss, metrics=[acc])

    train_steps = flags.FLAGS.steps_per_epoch
    val_steps = int(math.ceil(872 / 32))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss",
                                         patience=5,
                                         restore_best_weights=True),
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda _, __: checkpoint.save(checkpoint_prefix)),
        tf.keras.callbacks.TensorBoard(log_dir=job_dir),
    ]

    model.fit(train_data,
              epochs=100,
              steps_per_epoch=train_steps,
              validation_data=val_data,
              validation_steps=val_steps,
              initial_epoch=initial_epoch,
              callbacks=callbacks)


if __name__ == "__main__":
    app.flags.DEFINE_integer("steps_per_epoch", int(math.ceil(67349 / 32)),
                             "steps_per_epoch")
    app.flags.DEFINE_string("tfds_data_dir", "~/tensorflow_datasets",
                            "tfds_data_dir")
    app.flags.DEFINE_string("job-dir", "./lstm", "job-dir")

    app.run(main)
