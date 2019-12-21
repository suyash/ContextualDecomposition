import tensorflow as tf
from tensorflow.keras import Model  # pylint: disable=import-error
from tensorflow.keras.layers import Dense, Embedding, Input, LSTMCell, RNN  # pylint: disable=import-error

from .cd import lstm_decomposition


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


if __name__ == "__main__":
    tf.test.main()
