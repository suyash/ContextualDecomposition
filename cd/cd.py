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
