import numpy as np
from numpy import unique
from matplotlib import pyplot as plt
from scipy.stats import entropy
from scipy.misc import logsumexp
from itertools import product


# NOT tested
def data_model_kldiv(data, weights, visbias, hidbias, axis=0, serial=False):
    """
    Computes D_KL(Data Sample||Model).
    """
    # finding patterns and their logprobabilities in the data
    patterns, data_freq = unique(data, axis=axis, return_counts=True)
    data_freq = 1.*data_freq / data_freq.sum()
    # logprobs for the patterns in the data, from the RBM parameters
    rbm_logprobs = rbm_pattern_logprob(
        patterns, visbias, hidbias, weights, serial)
    rbm_probs = np.exp(rbm_logprobs)
    return entropy(data_freq, rbm_probs)


def exact_logZ_serial(bias1, bias2, weights):
    if bias1.size < bias2.size:
        bias1, bias2, weights = bias2, bias1, weights.T
    num = np.size(bias2)
    z = 0.
    for pattern in product([True, False], repeat=num):
        z += np.exp(rbm_pattern_logprob_unnorm(
                    [pattern], weights.T, bias1, bias2))
    return np.log(z)[0]


# tested, from write_probs.py
def exact_logZ(bias1, bias2, weights):
    if bias1.size < bias2.size:
        bias1, bias2, weights = bias2, bias1, weights.T
    num = np.size(bias2)
    all_patterns = np.array(list(product([True, False], repeat=num)))
    all_logprob = rbm_pattern_logprob_unnorm(
                all_patterns, weights.T, bias1, bias2)
    return logsumexp(all_logprob)


# tested, from write_probs.py
def rbm_pattern_logprob_unnorm(patterns, weights, bias1, bias2):
    r = np.dot(patterns, weights.T)
    r += bias1
    r = np.sum(np.log1p(np.exp(r)), axis=1)
    r += np.dot(patterns, bias2)
    return r


# from write_probs.py + serial
def rbm_pattern_logprob(patterns, visbias, hidbias, weights, serial=False):
    if serial:
        logZ = exact_logZ_serial(visbias, hidbias, weights)
    else:
        logZ = exact_logZ(visbias, hidbias, weights)
    return rbm_pattern_logprob_unnorm(
            patterns, weights, hidbias, visbias) - logZ


# from write_probs.py
def data_pattern_logprob(data, axis=0):
    patterns, data_freq = unique(data, axis)
    data_freq = data_freq / data_freq.sum()
    data_logprobs = np.log(data_freq)
    return patterns, data_logprobs


# one-pass covariance, may be useful in the future
def gen_cov(g):
    mean, covariance = 0, 0
    for i, x in enumerate(g):
        diff = x - mean
        mean += diff/(i+1)
        covariance += np.outer(diff, diff) * i / (i+1)
    return covariance/i


# def rbm_fim(sample, nvis):
#     # it should accept either a sample given as an array
#     # for example one saved before, or a generator given
#     # by the 'sample' method. As a consequence, this is
#     # NOT OPTIMISED for when sample is a np.array.
#     def fimfunction_rbm(state):
#         vis = state[:nvis]
#         hid = state[nvis:]
#         prod = np.outer(vis, hid)
#         return np.hstack([vis, hid, np.ravel(prod)])
#     s = [fimfunction_rbm(x) for x in sample]
#     return np.cov(s, rowvar=0)


def rbm_fim_numpy(sample, nvis):
    sample = np.asarray(sample)
    nsamples = sample.shape[0]
    vis = sample[:, :nvis]
    hid = sample[:, nvis:]
    prod = vis[:, :, np.newaxis] * hid[:, np.newaxis, :]
    # print(prod.shape)
    s = np.hstack([vis, hid, prod.reshape((nsamples, -1))])
    return np.cov(s, rowvar=0)


def rbm_fim_lowmem(sample, nvis):
    # it should accept either a sample given as an array
    # for example one saved before, or a generator given
    # by the 'sample' method. As a consequence, this is
    # NOT OPTIMISED for when sample is a np.array.
    def fimfunction_rbm(state):
        vis = state[:nvis]
        hid = state[nvis:]
        prod = np.outer(vis, hid)
        return np.hstack([vis, hid, np.ravel(prod)])
    s = (fimfunction_rbm(x) for x in sample)
    return gen_cov(s)


def fim_eig(fim, nvis, return_eigenvectors=False):
    if return_eigenvectors:
        nhid = (fim.shape[0]-nvis)//(1+nvis)
        # print(nvis, nhid)
        val, vec = np.linalg.eigh(fim)
        vec = vec[:, ::-1]
        # print(vec[nvis+nhid:].shape)
        return val[::-1], [vec[:nvis], vec[nvis:nvis+nhid],
                           vec[nvis+nhid:].reshape([nvis, nhid, -1])]
    return np.linalg.eigvalsh(fim)[::-1]


def select_clusters(times, clusterids, choice):
    idx = np.in1d(clusterids, choice)
    return times[idx], clusterids[idx]


def select_times(times, clusterids, min, max):
    idx = np.logical_and(times >= min, times <= max)
    return times[idx], clusterids[idx]


def plot_raster(times, clusterids, ax=None):
    if ax is None:
        ax = plt.gca()
    ax.plot(times, clusterids, ',')


def bin_timeseries(times, ids, dt):
    labels, l_idx = np.unique(ids, return_inverse=True)
    tmax = int(np.max(times) // dt) + 1
    assert len(times) == len(ids)
    binned = np.zeros((tmax, len(labels)), dtype=bool)
    for i, t in enumerate(times):
        ID = l_idx[i]
        T = int(t//dt)
        binned[T, ID] = True
    return binned

# # slower method, for testing only
# def bin_timeseries2(times, ids, dt):
#     labels = np.unique(ids)
#     tmax = np.max(times) + dt
#     assert len(times) == len(ids)
#     bins = np.arange(0, tmax, dt)
#     binned = np.empty([len(labels), len(bins)-1], dtype=bool)
#     for i, l in enumerate(labels):
#         ts = times[ids == l]
#         binned[i] = np.histogram(ts, bins=bins)[0].astype(bool)
#     return binned.T


def sum_over(a, n):
    # the first dimension (0) must be time
    b = np.reshape(a, [a.shape[0]//n, n, a.shape[1]])
    return np.sum(b, axis=1)


def zipf(array, axis=0):
    ncounts = array.shape[axis]
    _, counts = unique(array, axis=axis, return_counts=True)
    counts.sort()
    return 1.*counts[::-1]/ncounts


def plot_zipf(array, axis=0, normalise=False, **kwargs):
    z = zipf(array, axis)
    if normalise:
        plt.loglog(np.arange(1, len(z)+1), z/z[0], **kwargs)
    else:
        plt.loglog(np.arange(1, len(z)+1), z, **kwargs)


def detect_avalanches(activity):
    a_sizes = []
    a_durations = []
    curr_a_duration, curr_a_size = 0, 0

    for a in activity:
        if a == 0:
            if curr_a_duration > 0:
                a_sizes.append(curr_a_size)
                a_durations.append(curr_a_duration)
            curr_a_size = 0
            curr_a_duration = 0
            continue
        curr_a_duration += 1
        curr_a_size += a
    return a_sizes, a_durations
