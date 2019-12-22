import tensorflow as tf
from tensorflow.keras import Model  # pylint: disable=import-error
from tensorflow.keras.layers import Conv1D, Dense, Embedding, Input, LSTMCell, MaxPool1D, RNN  # pylint: disable=import-error

from .cd import conv1d_decomposition, lstm_decomposition, max_pool1d_decomposition


class LSTMDecompositionTest(tf.test.TestCase):
    def setUp(self):
        seq = Input((None, ), dtype="int32")
        x = Embedding(8000, 128)(seq)
        self.embedder = Model(seq, x)

        x = RNN(LSTMCell(128))(x)
        self.lstmer = Model(seq, x)

    def test_decomposition(self):
        inp = tf.random.uniform((10, 16),
                                minval=0,
                                maxval=8000,
                                dtype=tf.int32)

        lstm_inp = self.embedder(inp)
        desired_output = self.lstmer(inp)

        _, k, rk, b = self.lstmer.weights
        rel_h, irrel_h = lstm_decomposition(lstm_inp, k, rk, b, 5, 10)

        obtained_h = rel_h + irrel_h

        self.assertAllClose(obtained_h, desired_output)


class CNN1DDecompositionTest(tf.test.TestCase):
    def setUp(self):
        seq = Input((None, 128))
        x = Conv1D(filters=64, kernel_size=3, activation=None)(seq)
        self.model = Model(seq, x)

    def test_decomposition(self):
        relevant = tf.random.uniform((1, 10, 128))
        irrelevant = tf.random.uniform((1, 10, 128))

        k, b = self.model.weights
        rel_output, irrel_output = conv1d_decomposition(
            relevant, irrelevant, k, b)

        desired_output = self.model(relevant + irrelevant)

        self.assertAllClose(rel_output + irrel_output,
                            desired_output,
                            atol=1e-5,
                            rtol=1e-5)


class MaxPool1DDecompositionTest(tf.test.TestCase):
    def setUp(self):
        seq = Input((None, 128))
        x = MaxPool1D(pool_size=2)(seq)
        self.model = Model(seq, x)

    def test_decomposition(self):
        relevant = tf.random.uniform((1, 10, 128))
        irrelevant = tf.random.uniform((1, 10, 128))

        desired_output = self.model(relevant + irrelevant)

        rel_output, irrel_output = max_pool1d_decomposition(relevant,
                                                            irrelevant,
                                                            ksize=2)

        self.assertAllClose(rel_output + irrel_output, desired_output)


if __name__ == "__main__":
    tf.test.main()
