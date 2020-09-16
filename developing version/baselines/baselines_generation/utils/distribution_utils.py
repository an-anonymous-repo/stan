"""Distribution utility functions."""

def get_distribution_with_laplace_smoothing(a_count):
    k = 1.0
    tot_k = len(a_count) * k
    sum_count = sum(a_count)
    p = []
    for component in a_count:
        adj_component = (component + k) / (sum_count + tot_k)
        p.append(adj_component)
    # print('laplace_smoothing:\n', len(a_count), sum(a_count), sum(p), max(p), min(p))
    return p

def get_distribution(a_count):
    sum_count = sum(a_count)
    p = []
    for component in a_count:
        adj_component = component / sum_count
        p.append(adj_component)
    # print('get_distribution:\n', len(a_count), sum(a_count), sum(p), max(p), min(p))
    return p

def log_sum_exp_trick(ns):
    import numpy as np
    max = np.max(ns)
    ds = ns - max
    sumOfExp = np.exp(ds).sum()
    return max + np.log(sumOfExp)