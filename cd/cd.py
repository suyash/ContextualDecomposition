import tensorflow as tf


def lstm_decomposition(x, kernel, recurrent_kernel, bias, q, r):
    """
    x: [None, None, d_model]
    kernel: [d_emb, d_model * 4]
    recurrent_kernel: [d_model, d_model * 4]
    bias: [d_model,]
    q: int32
    r: int32
    """
    k_i, k_f, k_c, k_o = tf.split(kernel, num_or_size_splits=4, axis=1)
    rk_i, rk_f, rk_c, rk_o = tf.split(recurrent_kernel,
                                      num_or_size_splits=4,
                                      axis=1)
    b_i, b_f, b_c, b_o = tf.split(bias, num_or_size_splits=4, axis=0)

    relevant_h = tf.zeros((tf.shape(x)[0], tf.shape(x)[-1]))
    irrelevant_h = tf.zeros((tf.shape(x)[0], tf.shape(x)[-1]))
    relevant_c = tf.zeros((tf.shape(x)[0], tf.shape(x)[-1]))
    irrelevant_c = tf.zeros((tf.shape(x)[0], tf.shape(x)[-1]))

    for pos in range(tf.shape(x)[1]):
        inputs = x[:, pos, :]

        rel_i = tf.matmul(relevant_h, rk_i)
        rel_f = tf.matmul(relevant_h, rk_f)
        rel_c = tf.matmul(relevant_h, rk_c)
        rel_o = tf.matmul(relevant_h, rk_o)

        irrel_i = tf.matmul(irrelevant_h, rk_i)
        irrel_f = tf.matmul(irrelevant_h, rk_f)
        irrel_c = tf.matmul(irrelevant_h, rk_c)
        irrel_o = tf.matmul(irrelevant_h, rk_o)

        if pos >= q and pos <= r:
            rel_i = rel_i + tf.matmul(inputs, k_i)
            rel_c = rel_c + tf.matmul(inputs, k_c)
            rel_f = rel_f + tf.matmul(inputs, k_f)
            rel_o = rel_o + tf.matmul(inputs, k_o)
        else:
            irrel_i = irrel_i + tf.matmul(inputs, k_i)
            irrel_c = irrel_c + tf.matmul(inputs, k_c)
            irrel_f = irrel_f + tf.matmul(inputs, k_f)
            irrel_o = irrel_o + tf.matmul(inputs, k_o)

        rel_l_i, irrel_l_i, b_l_i = _decomp_three(rel_i, irrel_i, b_i,
                                                  tf.math.sigmoid)
        rel_l_c, irrel_l_c, b_l_c = _decomp_three(rel_c, irrel_c, b_c,
                                                  tf.math.tanh)

        relevant = rel_l_i * (rel_l_c + b_l_c) + b_l_i * rel_l_c
        irrelevant = irrel_l_i * (rel_l_c + irrel_l_c +
                                  b_l_c) + irrel_l_c * (rel_l_i + b_l_i)

        if pos >= q and pos <= r:
            relevant += b_l_i * b_l_c
        else:
            irrelevant += b_l_i * b_l_c

        if pos > 0:
            rel_l_f, irrel_l_f, b_l_f = _decomp_three(rel_f, irrel_f, b_f,
                                                      tf.math.sigmoid)
            relevant += (rel_l_f + b_l_f) * relevant_c
            irrelevant += (rel_l_f + irrel_l_f +
                           b_l_f) * irrelevant_c + irrel_l_f * relevant_c

        o = tf.math.sigmoid(
            tf.matmul(inputs, k_o) +
            tf.matmul(relevant_h + irrelevant_h, rk_o) + b_o)

        next_rel_h, next_irrel_h = _decomp_two(relevant, irrelevant,
                                               tf.math.tanh)

        next_rel_h = o * next_rel_h
        next_irrel_h = o * next_irrel_h

        relevant_h = next_rel_h
        irrelevant_h = next_irrel_h
        relevant_c = relevant
        irrelevant_c = irrelevant

    return relevant_h, irrelevant_h


def _decomp_three(a, b, c, f):
    a_contrib = 0.5 * (f(a + c) - f(c) + f(a + b + c) - f(b + c))
    b_contrib = 0.5 * (f(b + c) - f(c) + f(a + b + c) - f(a + c))
    return a_contrib, b_contrib, f(c)


def _decomp_two(a, b, f):
    return 0.5 * (f(a) + (f(a + b) - f(b))), 0.5 * (f(b) + (f(a + b) - f(a)))


def conv1d_decomposition(relevant, irrelevant, conv_kernel, bias):
    relevant = tf.nn.conv1d(relevant, conv_kernel, stride=1, padding="VALID")
    irrelevant = tf.nn.conv1d(irrelevant,
                              conv_kernel,
                              stride=1,
                              padding="VALID")

    relevant_abs = tf.math.abs(relevant)
    irrelevant_abs = tf.math.abs(irrelevant)

    relevant += bias * (relevant_abs / (relevant_abs + irrelevant_abs))
    irrelevant += bias * (irrelevant_abs / (relevant_abs + irrelevant_abs))

    return relevant, irrelevant


def act_decomposition(relevant, irrelevant, act_fn):
    return (act_fn(relevant), act_fn(relevant + irrelevant) - act_fn(relevant))


def max_pool1d_decomposition(relevant, irrelevant, ksize):
    relevant = tf.expand_dims(relevant, 1)
    irrelevant = tf.expand_dims(irrelevant, 1)

    o, a = tf.nn.max_pool_with_argmax(relevant + irrelevant,
                                      ksize=(1, ksize),
                                      strides=ksize,
                                      padding="VALID")

    relevant = tf.gather(
        params=tf.reshape(relevant, [tf.size(relevant)]),
        indices=tf.reshape(a, [tf.size(a)]),
    )

    irrelevant = tf.gather(
        params=tf.reshape(irrelevant, [tf.size(irrelevant)]),
        indices=tf.reshape(a, [tf.size(a)]),
    )

    relevant = tf.reshape(relevant, tf.shape(o))
    irrelevant = tf.reshape(irrelevant, tf.shape(o))

    relevant = tf.squeeze(relevant, [1])
    irrelevant = tf.squeeze(irrelevant, [1])

    return relevant, irrelevant


def cnn_net_decomposition(x, conv_weights, q, r):
    a, b, c = tf.shape(x)

    relevant = tf.concat(
        [
            tf.zeros((a, q, c)),
            x[:, q:(r + 1), :],
            tf.zeros((a, b - (r + 1), c)),
        ],
        axis=1,
    )

    irrelevant = tf.concat(
        [
            x[:, :q, :],
            tf.zeros((a, (r + 1) - q, c)),
            x[:, (r + 1):, :],
        ],
        axis=1,
    )

    rel_outputs, irrel_outputs = [], []

    for w in conv_weights:
        rel, irrel = conv1d_decomposition(relevant, irrelevant, w[0], w[1])
        rel, irrel = act_decomposition(rel, irrel, act_fn=tf.math.tanh)
        rel, irrel = max_pool1d_decomposition(rel, irrel, ksize=rel.shape[1])
        rel, irrel = tf.squeeze(rel, axis=[1]), tf.squeeze(irrel, axis=[1])

        rel_outputs.append(rel)
        irrel_outputs.append(irrel)

    relevant = tf.concat(rel_outputs, axis=-1)
    irrelevant = tf.concat(irrel_outputs, axis=-1)

    return relevant, irrelevant
