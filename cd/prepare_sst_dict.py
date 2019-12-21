from collections import defaultdict

from absl import app, flags, logging
import numpy as np
import tensorflow_datasets as tfds


def main(_):
    train_data = tfds.load("glue/sst2", split=tfds.Split.TRAIN)

    words = defaultdict(int)

    for d in train_data:
        s = d["sentence"]
        s = s.numpy().decode("utf-8")
        for w in s.split():
            words[w] += 1

    words = list(words.items())
    words = sorted(words, key=lambda x: x[1], reverse=True)
    logging.info("got %d words", len(words))

    x = 0
    while x < len(words):
        if words[x][1] == flags.FLAGS.min_freq:
            break
        x += 1

    logging.info("keeping %d words", x)

    words = words[:x]

    tokens = list(map(lambda x: x[0], words))

    np.save("cd/tokens.npy", tokens)


if __name__ == "__main__":
    app.flags.DEFINE_integer("min_freq", 2, "min_freq")

    app.run(main)
