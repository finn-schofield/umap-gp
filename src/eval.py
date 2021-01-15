from sklearn.metrics import pairwise_distances
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np
from math import isnan
from umap.umap_ import fuzzy_simplicial_set

SMOOTH_K_TOLERANCE = 1e-5
MIN_K_DIST_SCALE = 1e-3


def umap_cost(embedding, v):
    """

    :param embedding: low dimensional embedding of data
    :param v: the membership strength of the instances in the original dimensions
    :return:
    """
    w = calculate_w(pairwise_distances(embedding))
    # w = w[~np.eye(w.shape[0], dtype=bool)].reshape(w.shape[0], -1)  # Remove main diagonal
    # v = v[~np.eye(v.shape[0], dtype=bool)].reshape(v.shape[0], -1)
    # w = np.where(w == 1.0, w - 1e-4, w)
    # a = (np.multiply(v, np.log(w)))
    # b = np.multiply((1 - v), np.log(1 - w))

    a = v * np.log(w + 0.01)
    b = (1 - v) * np.log(1 - w + 0.01)

    cost = - np.sum(a + b)
    if isnan(cost):
        cost = np.inf
    return cost


def calculate_w(x, a=1.929, b=0.7915):
    # w = 1.0 / (1.0 + a * np.power(x, (2 * b)))
    w = np.power(1 + a * np.square(x)**b, -1)
    return w


def classif_eval(data, labels):
    clf = RandomForestClassifier()
    scores = cross_val_score(clf, data, labels, cv=10)
    return scores.mean()



