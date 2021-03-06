"""
For pure conv model:
    lr_decay_steps = 1000
    batch_size = 32
"""

import os

from absl import app, flags, logging
import tensorflow as tf
from tensorflow.keras import Model  # pylint: disable=import-error
from tensorflow.keras.layers import Activation, Conv2D, Dense, Flatten, Input, MaxPool2D, Reshape  # pylint: disable=import-error
import tensorflow_datasets as tfds

from .layers import Activation as ActivationDec, Conv2D as Conv2DDec, Dense as DenseDec, MaxPool2D as MaxPool2DDec


def prepare_dataset(d):
    return d.map(lambda a, b: (tf.cast(a, tf.float32) / 255., b))


def prepare_model():
    x = inp = Input((28, 28, 1))

    x = Conv2D(filters=16,
               kernel_size=(5, 5),
               strides=(1, 1),
               activation=None,
               kernel_regularizer=tf.keras.regularizers.l2(4e-5))(
                   x)  # 24x24x16
    x = Activation("tanh")(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)  # 12x12x16

    x = Conv2D(filters=32,
               kernel_size=(5, 5),
               strides=(1, 1),
               activation=None,
               kernel_regularizer=tf.keras.regularizers.l2(4e-5))(x)  # 8x8x32
    x = Activation("tanh")(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)  # 4x4x32

    x = Conv2D(filters=64,
               kernel_size=(3, 3),
               strides=(1, 1),
               activation=None,
               kernel_regularizer=tf.keras.regularizers.l2(4e-5))(x)  # 2x2x64
    x = Activation("tanh")(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)  # 1x1x64

    x = Flatten()(x)
    x = Dense(10,
              activation=None,
              kernel_regularizer=tf.keras.regularizers.l2(4e-5))(x)

    return Model(inp, x)


def prepare_pure_conv_model():
    x = inp = Input((28, 28, 1))

    x = Conv2D(
        filters=8,
        kernel_size=(5, 5),
        strides=(1, 1),
        activation=None,
        kernel_regularizer=tf.keras.regularizers.l2(4e-5),
    )(x)  # 24
    x = Activation("tanh")(x)

    x = Conv2D(
        filters=8,
        kernel_size=(5, 5),
        strides=(1, 1),
        activation=None,
        kernel_regularizer=tf.keras.regularizers.l2(4e-5),
    )(x)  # 20
    x = Activation("tanh")(x)

    x = Conv2D(
        filters=16,
        kernel_size=(5, 5),
        strides=(1, 1),
        activation=None,
        kernel_regularizer=tf.keras.regularizers.l2(4e-5),
    )(x)  # 16
    x = Activation("tanh")(x)

    x = Conv2D(
        filters=16,
        kernel_size=(5, 5),
        strides=(1, 1),
        activation=None,
        kernel_regularizer=tf.keras.regularizers.l2(4e-5),
    )(x)  # 12
    x = Activation("tanh")(x)

    x = Conv2D(
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        activation=None,
        kernel_regularizer=tf.keras.regularizers.l2(4e-5),
    )(x)  # 8
    x = Activation("tanh")(x)

    x = Conv2D(
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        activation=None,
        kernel_regularizer=tf.keras.regularizers.l2(4e-5),
    )(x)  # 4
    x = Activation("tanh")(x)

    x = Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=(1, 1),
        activation=None,
        kernel_regularizer=tf.keras.regularizers.l2(4e-5),
    )(x)  # 2
    x = Activation("tanh")(x)

    x = Conv2D(
        filters=64,
        kernel_size=(2, 2),
        strides=(1, 1),
        activation=None,
        kernel_regularizer=tf.keras.regularizers.l2(4e-5),
    )(x)  # 1
    x = Activation("tanh")(x)

    x = Reshape((64, ))(x)

    x = Dense(
        10,
        activation=None,
        kernel_regularizer=tf.keras.regularizers.l2(4e-5),
    )(x)

    return Model(inp, x)


def prepare_decomp_model():
    rel = reli = Input((28, 28, 1), name="rel")
    irrel = irreli = Input((28, 28, 1), name="irrel")

    rel, irrel = Conv2DDec(filters=16, kernel_size=(5, 5),
                           strides=(1, 1))([rel, irrel])
    rel, irrel = ActivationDec("tanh")([rel, irrel])
    rel, irrel = MaxPool2DDec(pool_size=(2, 2), strides=(2, 2))([rel, irrel])

    rel, irrel = Conv2DDec(filters=32, kernel_size=(5, 5),
                           strides=(1, 1))([rel, irrel])
    rel, irrel = ActivationDec("tanh")([rel, irrel])
    rel, irrel = MaxPool2DDec(pool_size=(2, 2), strides=(2, 2))([rel, irrel])

    rel, irrel = Conv2DDec(filters=64, kernel_size=(3, 3),
                           strides=(1, 1))([rel, irrel])
    rel, irrel = ActivationDec("tanh")([rel, irrel])
    rel, irrel = MaxPool2DDec(pool_size=(2, 2), strides=(2, 2))([rel, irrel])

    rel = Flatten()(rel)
    irrel = Flatten()(irrel)

    rel, irrel = DenseDec(10)([rel, irrel])

    return Model([reli, irreli], [rel, irrel])


def train(epochs, early_stopping_patience, pure_conv_model,
          lr_schedule_decay_rate, lr_schedule_decay_steps, batch_size,
          data_dir, job_dir):
    datasets = tfds.load("mnist", as_supervised=True, data_dir=data_dir)

    train_data = prepare_dataset(datasets[tfds.Split.TRAIN]) \
                .cache() \
                .prefetch(tf.data.experimental.AUTOTUNE) \
                .shuffle(buffer_size=1000) \
                .batch(batch_size) \
                .repeat()

    test_data = prepare_dataset(datasets[tfds.Split.TEST]) \
                .cache() \
                .prefetch(tf.data.experimental.AUTOTUNE) \
                .batch(batch_size)

    model = prepare_model()
    model.summary()

    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=1e-3,
        decay_steps=lr_schedule_decay_steps,
        decay_rate=lr_schedule_decay_rate)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    acc = tf.keras.metrics.SparseCategoricalAccuracy()

    model.compile(optimizer=optimizer, loss=loss, metrics=[acc])

    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=job_dir)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_sparse_categorical_accuracy",
        patience=early_stopping_patience,
        verbose=1,
        restore_best_weights=True)

    model.fit(train_data,
              epochs=epochs,
              steps_per_epoch=60000 // 25,
              validation_data=test_data,
              validation_steps=10000 // 25,
              callbacks=[tensorboard, early_stopping])

    model.save(os.path.join(job_dir, "saved_model"))


def main(_):
    tf.random.set_seed(42)
    train(epochs=flags.FLAGS.epochs,
          early_stopping_patience=flags.FLAGS.early_stopping_patience,
          pure_conv_model=flags.FLAGS.pure_conv_model,
          lr_schedule_decay_rate=flags.FLAGS.lr_schedule_decay_rate,
          lr_schedule_decay_steps=flags.FLAGS.lr_schedule_decay_steps,
          batch_size=flags.FLAGS.batch_size,
          data_dir=flags.FLAGS.tfds_data_dir,
          job_dir=flags.FLAGS["job-dir"].value)


if __name__ == "__main__":
    app.flags.DEFINE_integer("epochs", 100, "epochs")
    app.flags.DEFINE_integer("early_stopping_patience", 10,
                             "early_stopping_patience")
    app.flags.DEFINE_boolean("pure_conv_model", False,
                             "train model not containing pooling")
    app.flags.DEFINE_float("lr_schedule_decay_rate", 0.96,
                           "lr_schedule_decay_rate")
    app.flags.DEFINE_integer("lr_schedule_decay_steps", 10000,
                             "lr_schedule_decay_steps")
    app.flags.DEFINE_integer("batch_size", 25, "batch_size")
    app.flags.DEFINE_string("tfds_data_dir", "~/tensorflow_datasets",
                            "tfds_data_dir")
    app.flags.DEFINE_string("job-dir", "./1", "job-dir")

    app.run(main)
