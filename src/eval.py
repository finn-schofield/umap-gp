from sklearn.metrics import pairwise_distances
import numpy as np
from umap.umap_ import fuzzy_simplicial_set

SMOOTH_K_TOLERANCE = 1e-5
MIN_K_DIST_SCALE = 1e-3


def umap_cost(data, embedding, v, k=15):
    """

    :param data: found embedding
    :param embedding: umap embedding to be optimised against
    :param k: nearest neighbors to check for each instance
    :return:
    """

    w = calculate_w(pairwise_distances(embedding))
    a = 0
    b = 0
    cost = - ((v * np.log(w)) + np.where(w == 1.0, w, ((1 - v) * np.log(1 - w))))
    # for i in range(data.shape[0]):
    #     for j in range(data.shape[0]):
    #         a += v[i, j] * np.log(w[i, j])
    #         if w[i, j] == 1.0:
    #             b += 1.0
    #         else:
    #             b += (1 - v[i, j]) * np.log(1 - w[i, j])
    #
    # cost = - (a + b)
    return cost


def calculate_w(x, a=1.929, b=0.7915):
    return 1.0 / (1.0 + a * np.power(x, (2 * b)))



