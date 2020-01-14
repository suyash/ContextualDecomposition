import functools

import numpy as np
from skimage import measure
import tensorflow as tf

from .cd import lstm_decomposition, cnn_net_decomposition


def create_mask(scores):
    a = np.abs(scores)
    s = np.sum(~np.isnan(a))

    if s == 5:
        return a > np.nanpercentile(a, 59)

    if s == 4:
        return a > np.nanpercentile(a, 49)

    if s == 3:
        return a > np.nanpercentile(a, 59)

    if s == 2:
        return a > np.nanpercentile(a, 49)

    if s == 1:
        return ~np.isnan(a)

    return a > np.nanpercentile(a, 99.5)


def acd_1d_decomposition(score, l):
    """
    score: f(int, int) -> float32
    l: int
    """

    scores = []
    for i in range(l):
        scores.append(score(i, i))
    scores = np.array(scores)

    m = np.array([False] * l)
    comps_all = [np.array([0] * l)]
    comp_scores_all = [np.copy(scores).tolist()]

    while True:
        m = m | create_mask(scores)
        comps = measure.label(m)
        n_comp = np.max(comps)
        comp_scores = [0] * n_comp

        for i in range(1, n_comp + 1):
            cm = comps == i

            start = np.argmax(cm)
            end = start + np.argmin(cm[start:])
            if end == start:
                end = cm.size

            score_current = score(start, end - 1)
            comp_scores[i - 1] = score_current

            if start > 0:
                score_l = score(start - 1, end - 1)
                scores[start - 1] = score_l - score_current

            if end < cm.size:
                score_r = score(start, end)
                scores[end] = score_r - score_current

        comps_all.append(comps)
        comp_scores_all.append(comp_scores)

        scores[m] = np.nan

        if np.sum(np.isnan(scores)) == scores.size:
            break

    return comps_all, comp_scores_all


def lstm_score(p, q, embed_inp, k, rk, b, dw, db):
    rel, _ = lstm_decomposition(embed_inp, k, rk, b, p, q)
    pred = tf.matmul(rel, dw) + db
    pred = pred.numpy()
    return pred[0, 1] - pred[0, 0]


def lstm_acd_decomposition(inp, model):
    """
    inp: tf.Tensor(dtype=np.int32, shape=(1, -1)) tokenized input
    model: tf.keras.Model or equivalent
    """

    l = inp.numpy().size
    e, k, rk, b, dw, db = model.weights

    embed_inp = tf.nn.embedding_lookup(params=e, ids=inp)

    return acd_1d_decomposition(
        functools.partial(lstm_score,
                          embed_inp=embed_inp,
                          k=k,
                          rk=rk,
                          b=b,
                          dw=dw,
                          db=db), l)


def agglomerate_acd_1d_decomposition(comps, comp_scores):
    cache = {}
    maxlevel = 0

    for i in range(len(comps[0])):
        cache[(i, i + 1)] = (0, comp_scores[0][i], [])

    for i in range(1, len(comps)):
        c = comps[i]
        cs = comp_scores[i]

        n_comps = np.max(c)

        for j in range(1, n_comps + 1):
            m = c == j

            start = np.argmax(m)
            end = start + np.argmin(m[start:])
            if end == start:
                end = m.size

            csize = end - start

            if csize == 1:
                pass  # already handled
            elif i > 0:
                aff = comps[i - 1][m]
                matches = np.unique(aff)
                n_matches = matches.size

                if n_matches == 0:
                    cache[(start, end)] = (1, cs[j - 1], [])
                elif n_matches == 1:
                    if (start, end) not in cache:
                        cache[(start, end)] = (1, cs[j - 1], [])
                else:
                    aff_comps = measure.label(aff, background=-1)
                    n_aff_comps = np.max(aff_comps)
                    children = []
                    ml = 0
                    for match in range(1, n_aff_comps + 1):
                        mm = aff_comps == match
                        aff_start = start + np.argmax(mm)
                        aff_end = aff_start + np.sum(mm)
                        children.append((aff_start, aff_end))
                        ml = max(ml, cache[(aff_start, aff_end)][0])

                    cache[(start, end)] = (ml + 1, cs[j - 1], children)
                    maxlevel = max(maxlevel, ml + 1)

    tree = [[] for i in range(maxlevel + 1)]
    for k, v in cache.items():
        tree[v[0]].append((k, v[1], v[2]))

    return tree


def conv1d_score(p, q, embed_inp, conv_weights, dk, db):
    rel, _ = cnn_net_decomposition(embed_inp, conv_weights, p, q)
    pred = tf.matmul(rel, dk) + db
    pred = pred.numpy()
    return pred[0, 1] - pred[0, 0]


def conv1d_acd_decomposition(inp, model):
    """
    inp: tf.Tensor(dtype=np.int32, shape=(1, -1)) tokenized input
    model: tf.keras.Model or equivalent
    """

    l = inp.numpy().size
    weights = model.weights

    embed_inp = tf.nn.embedding_lookup(params=weights[0], ids=inp)

    conv_weights = []
    for i in range((len(weights) - 3) // 2):
        conv_weights.append([weights[2 * i + 1], weights[2 * i + 2]])

    return acd_1d_decomposition(
        functools.partial(conv1d_score,
                          embed_inp=embed_inp,
                          conv_weights=conv_weights,
                          dk=weights[-2],
                          db=weights[-1]), l)
