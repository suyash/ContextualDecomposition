import tensorflow as tf
from tensorflow.keras import Model  # pylint: disable=import-error
from tensorflow.keras.layers import Input, Conv2D, Dense, MaxPool2D  # pylint: disable=import-error

from .layers import Conv2D as Conv2DDec, MaxPool2D as MaxPool2DDec, Dense as DenseDec


class Conv2DTest(tf.test.TestCase):
    def setUp(self):
        i = Input((None, None, 3))
        x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1))(i)
        self.m1 = Model(i, x)

        r = Input((None, None, 3))
        ir = Input((None, None, 3))
        x_r, x_ir = Conv2DDec(filters=16, kernel_size=(3, 3),
                              strides=(1, 1))([r, ir])
        self.m2 = Model([r, ir], [x_r, x_ir])

        self.m2.set_weights(self.m1.get_weights())

    def test_conv2d(self):
        r = tf.random.uniform((8, 16, 16, 3))
        ir = tf.random.uniform((8, 16, 16, 3))

        obtained_r, obtained_ir = self.m2([r, ir])
        desired = self.m1(r + ir)

        self.assertAllClose(desired, obtained_r + obtained_ir)


class MaxPool2DTest(tf.test.TestCase):
    def setUp(self):
        i = Input((None, None, 3))
        x = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(i)
        self.m1 = Model(i, x)

        r = Input((None, None, 3))
        ir = Input((None, None, 3))
        x_r, x_ir = MaxPool2DDec(pool_size=(2, 2), strides=(2, 2))([r, ir])
        self.m2 = Model([r, ir], [x_r, x_ir])

        self.m2.set_weights(self.m1.get_weights())

    def test_max_pool_2d(self):
        r = tf.random.uniform((8, 16, 16, 3))
        ir = tf.random.uniform((8, 16, 16, 3))

        obtained_r, obtained_ir = self.m2([r, ir])
        desired = self.m1(r + ir)

        self.assertAllClose(desired, obtained_r + obtained_ir)


class DenseTest(tf.test.TestCase):
    def setUp(self):
        i = Input((128, ))
        x = Dense(128)(i)
        self.m1 = Model(i, x)

        r = Input((128, ))
        ir = Input((128, ))
        x_r, x_ir = DenseDec(128)([r, ir])
        self.m2 = Model([r, ir], [x_r, x_ir])

        self.m2.set_weights(self.m1.get_weights())

    def test_dense(self):
        r = tf.random.uniform((8, 128))
        ir = tf.random.uniform((8, 128))

        obtained_r, obtained_ir = self.m2([r, ir])
        desired = self.m1(r + ir)

        self.assertAllClose(desired, obtained_r + obtained_ir)


if __name__ == "__main__":
    tf.test.main()
