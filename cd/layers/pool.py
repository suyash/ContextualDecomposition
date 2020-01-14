import tensorflow as tf
from tensorflow.keras.layers import Layer  # pylint: disable=import-error


class MaxPool2D(Layer):
    def __init__(self, pool_size, strides, **kwargs):
        super().__init__(**kwargs)

        self.pool_size = pool_size
        self.strides = strides

    def call(self, x):
        rel, irrel = x

        o, a = tf.nn.max_pool_with_argmax(input=(rel + irrel),
                                          ksize=self.pool_size,
                                          strides=self.strides,
                                          padding="VALID")

        rel = tf.reshape(rel, (tf.shape(rel)[0], -1))
        irrel = tf.reshape(irrel, (tf.shape(irrel)[0], -1))
        a = tf.reshape(a, (tf.shape(a)[0], -1))

        rel = tf.gather(params=rel, indices=a, batch_dims=1)
        irrel = tf.gather(params=irrel, indices=a, batch_dims=1)

        rel = tf.reshape(rel, tf.shape(o))
        irrel = tf.reshape(irrel, tf.shape(o))

        return rel, irrel
