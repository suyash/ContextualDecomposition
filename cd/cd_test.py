import tensorflow as tf
from tensorflow.keras import Model  # pylint: disable=import-error
from tensorflow.keras.layers import Conv1D, Dense, Embedding, Input, LSTMCell, MaxPool1D, RNN  # pylint: disable=import-error

from .cd import cnn_net_decomposition, conv1d_decomposition, lstm_decomposition, max_pool1d_decomposition
from .cnn import prepare_model


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


class CNNDecomposition(tf.test.TestCase):
    def setUp(self):
        m = prepare_model(64, 4e-6, 8000, [(64, 2), (64, 3), (64, 4), (64, 5)])
        self.model = Model(m.inputs, m.layers[-2].output)

    def test_decomposition(self):
        x = tf.random.uniform((1, 32), minval=0, maxval=8000, dtype=tf.int32)

        w = self.model.weights

        conv_weights = []
        for i in range((len(w) - 1) // 2):
            conv_weights.append((w[2 * i + 1], w[2 * i + 2]))

        inp = tf.nn.embedding_lookup(params=w[0], ids=x)
        rel, irrel = cnn_net_decomposition(inp, conv_weights, q=5, r=10)

        desired_output = self.model(x)

        self.assertAllClose(desired_output, rel + irrel)


if __name__ == "__main__":
    tf.test.main()
