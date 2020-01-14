import tensorflow as tf
from tensorflow.keras import Model  # pylint: disable=import-error
from tensorflow.keras.layers import Dense, Embedding, Input, LSTM  # pylint: disable=import-error

from .acd import lstm_acd_decomposition


class LSTMACDDecompositionTest(tf.test.TestCase):
    def setUp(self):
        seq = Input((None, ), dtype="int32")
        x = Embedding(8000, 128)(seq)
        x = LSTM(128)(x)
        x = Dense(2, activation=None)(x)
        self.model = Model(seq, x)

    def test_acd_decomposition(self):
        inp = tf.random.uniform((1, 16),
                                minval=0,
                                maxval=8000,
                                dtype=tf.int32)

        _, comp_scores = lstm_acd_decomposition(inp, self.model)

        obtained_score = comp_scores[-1][0]

        pred = self.model.predict(inp)

        desired_score = pred[0, 1] - pred[0, 0]

        self.assertAllClose(desired_score, obtained_score)


if __name__ == "__main__":
    tf.test.main()
