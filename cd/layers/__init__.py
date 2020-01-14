import tensorflow as tf
from tensorflow.keras.layers import Layer  # pylint: disable=import-error

from .conv import Conv2D
from .pool import MaxPool2D


class Dense(Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)

        self.units = units
        self.bias_epsilon = 1e-16

    def build(self, input_shape):
        u = input_shape[0][-1]
        self.kernel = self.add_weight(shape=(u, self.units),
                                      initializer="glorot_uniform",
                                      name="kernel")
        self.bias = self.add_weight(shape=(self.units, ),
                                    initializer="zeros",
                                    name="bias")

    def call(self, x):
        rel, irrel = x

        w_r = tf.matmul(rel, self.kernel)
        w_ir = tf.matmul(irrel, self.kernel)

        w_r_a = tf.math.abs(w_r)
        w_ir_a = tf.math.abs(w_ir)
        den = w_r_a + w_ir_a + self.bias_epsilon
        b_r = (w_r_a / den) * self.bias
        b_ir = (w_ir_a / den) * self.bias

        return (w_r + b_r, w_ir + b_ir)


class Activation(Layer):
    def __init__(self, act, **kwargs):
        super().__init__(**kwargs)

        if isinstance(act, str):
            assert act == "tanh" or act == "relu"

            if act == "tanh":
                self.act = tf.math.tanh
            else:
                self.act = tf.nn.relu
        else:
            self.act = act

    def call(self, x):
        rel, irrel = x

        o = self.act(rel + irrel)
        rel = self.act(rel)

        return rel, o - rel
