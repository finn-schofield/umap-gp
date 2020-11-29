from sklearn.metrics import pairwise_distances
import numpy as np
from umap.umap_ import fuzzy_simplicial_set

SMOOTH_K_TOLERANCE = 1e-5
MIN_K_DIST_SCALE = 1e-3


def umap_cost(data, embedding, k=15):
    """

    :param data: found embedding
    :param embedding: umap embedding to be optimised against
    :param k: nearest neighbors to check for each instance
    :return:
    """

    v = fuzzy_simplicial_set(
        data,
        k,
        np.random.RandomState(1),
        "euclidean"
    )[0]
    w = calculate_w(pairwise_distances(embedding))

    print(v.shape)
    print(w.shape)
    a = 0
    b = 0
    cv = 0
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            cv += v[i, j] * np.log(v[i, j]) + (1 - v[i, j]) * np.log(1 - v[i, j])
            a += v[i, j] * np.log(w[i, j])
            if w[i, j] == 1.0:
                b += 1.0
            else:
                b += (1 - v[i, j]) * np.log(1 - w[i, j])

    cost = cv - a - b

    return cost


def calculate_w(x, a=1.929, b=0.7915):
    return 1.0 / (1.0 + a * np.power(x, (2 * b)))



