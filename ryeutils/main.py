from collections import Counter
import numpy as np
from itertools import groupby


def np_pearson_corr(x, y):
    """ Computes correlation between the rows/columns of 2 arrays.

    Copied from https://cancerdatascience.org/blog/sposts/pearson-correlation/

    Args:
        x (np.ndarray):
        y (np.ndarray):

    Returns:
        np.ndarray: returns 2d array, where the value at (i, j) = correlation(x[i], y[j])

    
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


def find_runs(x):
    run_values = []
    run_lengths = []
    groups = []
    # groups.append(list(g))      # Store group iterator as a list

    for key, group in groupby(x):
        g = list(group)
        groups.append(g)
        run_values.append(key)
        run_lengths.append(len(g))

    return run_values, run_lengths


def index_stimuli(stim_list, include_trial_idx=True):
    """
    Computes indices for list of stimuli.

    Example with stim_list = 'AAABBBCCCAAABBB':
      -       stim = AAA BBB CCC AAA BBB
      -   stim_occ = 012 012 012 345 345
      -    run_idx = 000 111 222 333 444
      - idx_in_run = 012 012 012 012 012
      -    run_occ = 000 111 222 333 444

    Args:
        stim_list (List[str]): List of stimuli
        include_trial_idx ():

    Returns:
        dict: contains keys ['stim', 'stim_occ', 'run_idx', 'idx_in_run', 'run_occ']

    """
    stimrun, stimrun_len = find_runs(stim_list)
    n_runs = len(stimrun)
    stimrun_occ = occurrence(stimrun)

    stim_occ = occurrence(stim_list)
    run_idx = [i for i in range(n_runs) for j in range(stimrun_len[i])]
    idx_in_run = [j for i in range(n_runs) for j in range(stimrun_len[i])]
    run_occ = [x for x, y in zip(stimrun_occ, stimrun_len) for i in range(y)]

    return dict(stim=stim_list,
                stim_occ=stim_occ,
                run_idx=run_idx,
                idx_in_run=idx_in_run,
                run_occ=run_occ)


def get_run_limits(stim_list):
    labels, runs = find_runs(stim_list)
    end_locs = np.cumsum(runs)
    start_locs = end_locs - np.array(runs)
    return labels, runs, end_locs, start_locs

# def get_subplot_label_locs(label_list):
#     stim_map = index_stimuli(label_list)
#     stim_run_list = [(x, y) for x, y in zip(stim_map['stim'], stim_map['run_idx'])]

# %%
