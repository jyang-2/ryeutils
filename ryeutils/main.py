from collections import Counter
import numpy as np


def np_pearson_corr(x, y):
    """ Computes correlation between the rows/columns of 2 arrays.

    Copied from https://cancerdatascience.org/blog/posts/pearson-correlation/
    """
    xv = x - x.mean(axis=0)
    yv = y - y.mean(axis=0)
    xvss = (xv * xv).sum(axis=0)
    yvss = (yv * yv).sum(axis=0)
    result = np.matmul(xv.transpose(), yv) / np.sqrt(np.outer(xvss, yvss))
    # bound the values to -1 to 1 in the event of precision issues
    return np.maximum(np.minimum(result, 1.0), -1.0)


def occurrence(x):
    """ Maps list elements to the nth occurrence of the element value in the list."""

    if isinstance(x, list):
        occ = _occurrence_list(x)

    elif isinstance(x, np.ndarray):
        occ = _occurrence_np(x)

    return occ


def _occurrence_np(x):
    counter = Counter()

    occ = np.zeros_like(x, dtype='int')

    for index, ele in np.ndenumerate(x):
        occ[index] = counter[ele]
        counter[ele] = counter[ele] + 1

    return occ


def _occurrence_list(x):
    counter = Counter()
    occ = []

    for item in x:
        occ.append(counter[item])
        counter[item] = counter[item] + 1

    return occ
#%%
