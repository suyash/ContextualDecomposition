from absl import app, flags, logging
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from .lstm import create_table, preprocess_dataset


def main(_):
    model = tf.keras.models.load_model(flags.FLAGS.model_dir)

    datasets = tfds.load("glue/sst2")

    tokens = np.load("cd/tokens.npy")
    table = create_table(tokens)

    val_data = preprocess_dataset(datasets[tfds.Split.VALIDATION], table)
    val_data = val_data.batch(1)

    _, a = model.evaluate(val_data)
    logging.info("Validation Accuracy: %f", a)

    # NOTE: All the labels in test set are -1?
    #
    # test_data = preprocess_dataset(datasets[tfds.Split.TEST], table)
    # test_data = test_data.map(lambda a, b: (a, 0))
    # test_data = test_data.batch(1)

    # _, a = model.evaluate(test_data)
    # logging.info("Test Accuracy: %f", a)


if __name__ == "__main__":
    app.flags.DEFINE_string("model_dir", "runs/lstm/21/saved_model/best",
                            "model_dir")

    app.run(main)
