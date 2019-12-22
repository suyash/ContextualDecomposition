import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def create_table(tokens):
    keys = tokens
    values = 1 + tf.range(len(keys))
    return tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, values), 1 + len(keys))


def create_inv_table(tokens):
    values = tokens
    keys = 1 + tf.range(len(values))
    return tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(keys, values), "<UNK>")


def preprocess_dataset(dataset, table):
    dataset = dataset.map(lambda x: (x["sentence"], x["label"]))
    dataset = dataset.map(lambda a, b: (tf.map_fn(
        lambda x: table.lookup(x), tf.strings.split(a), dtype=tf.int32), b))
    return dataset


def prepare_datasets(datasets, tokens, batch_size, val_min_len=None):
    train_data = datasets[tfds.Split.TRAIN]
    val_data = datasets[tfds.Split.VALIDATION]

    table = create_table(tokens)

    train_data = preprocess_dataset(train_data, table)
    train_data = train_data.cache()
    train_data = train_data.prefetch(tf.data.experimental.AUTOTUNE)
    train_data = train_data.shuffle(buffer_size=1000)
    train_data = train_data.padded_batch(batch_size,
                                         padded_shapes=((None, ), ()))
    train_data = train_data.repeat()

    val_data = preprocess_dataset(val_data, table)
    if val_min_len != None:
        val_data = val_data.map(lambda a, b: (tf.cond(
            tf.shape(a)[0] < val_min_len, lambda: tf.pad(
                a, [[0, val_min_len - tf.shape(a)[0]]]), lambda: a), b))

    val_data = val_data.batch(1)

    return train_data, val_data
