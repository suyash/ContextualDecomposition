import tensorflow as tf
from tensorflow.keras.layers import Layer  # pylint: disable=import-error


class Conv2D(Layer):
    def __init__(self, filters, kernel_size, strides, **kwargs):
        super().__init__(**kwargs)

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.bias_epsilon = 1e-16

    def build(self, input_shape):
        input_filters = input_shape[0][-1]

        self.kernel = self.add_weight(shape=(self.kernel_size[0],
                                             self.kernel_size[1],
                                             input_filters, self.filters),
                                      initializer="glorot_uniform",
                                      name="kernel")
        self.bias = self.add_weight(shape=(self.filters, ),
                                    initializer="zeros",
                                    name="bias")

    def call(self, x):
        rel, irrel = x

        w_r = tf.nn.conv2d(input=rel,
                           filters=self.kernel,
                           strides=self.strides,
                           padding="VALID")
        w_ir = tf.nn.conv2d(input=irrel,
                            filters=self.kernel,
                            strides=self.strides,
                            padding="VALID")

        w_r_a = tf.math.abs(w_r)
        w_ir_a = tf.math.abs(w_ir)
        den = w_r_a + w_ir_a + self.bias_epsilon
        b_r = (w_r_a / den) * self.bias
        b_ir = (w_ir_a / den) * self.bias

        return (w_r + b_r, w_ir + b_ir)
