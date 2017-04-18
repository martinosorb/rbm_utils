import numpy as np
from matplotlib import pyplot as plt


def gen_cov(g):
    mean, covariance = 0, 0
    for i, x in enumerate(g):
        diff = x - mean
        mean += diff/(i+1)
        covariance += np.outer(diff, diff) * i / (i+1)
    return covariance/i


def rbm_fim(sample, nvis):
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
        print(nvis, nhid)
        val, vec = np.linalg.eigh(fim)
        vec = vec[:, ::-1]
        print(vec[nvis+nhid:].shape)
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


def sum_over():
    return NotImplemented


def zipf(array, axis=0):
    ncounts = array.shape[axis]
    _, counts = unique(array, axis=axis, return_counts=True)
    counts.sort()
    return counts[::-1]/ncounts


def plot_zipf(array, axis=0, **kwargs):
    z = zipf(array, axis)
    plt.loglog(np.arange(1, len(z)+1), z, **kwargs)


def unique(ar, return_index=False, return_inverse=False,
           return_counts=False, axis=None):
    "Function will be in Numpy soon, it has an axis= argument"
    ar = np.asanyarray(ar)
    if axis is None:
        return np.unique(ar, return_index, return_inverse, return_counts)
    if abs(axis) > ar.ndim:
        raise ValueError('Invalid axis kwarg specified for unique')

    ar = np.swapaxes(ar, axis, 0)
    orig_shape, orig_dtype = ar.shape, ar.dtype
    # Must reshape to a contiguous 2D array for this to work...
    ar = ar.reshape(orig_shape[0], -1)
    ar = np.ascontiguousarray(ar)

    if ar.dtype.char in (np.typecodes['AllInteger'] + 'S'):
        # Optimization inspired by <http://stackoverflow.com/a/16973510/325565>
        dtype = np.dtype((np.void, ar.dtype.itemsize * ar.shape[1]))
    else:
        dtype = [('f{i}'.format(i=i), ar.dtype) for i in range(ar.shape[1])]

    try:
        consolidated = ar.view(dtype)
    except TypeError:
        # There's no good way to do this for object arrays, etc...
        msg = 'The axis argument to unique is not supported for dtype {dt}'
        raise TypeError(msg.format(dt=ar.dtype))

    def reshape_uniq(uniq):
        uniq = uniq.view(orig_dtype)
        uniq = uniq.reshape(-1, *orig_shape[1:])
        uniq = np.swapaxes(uniq, 0, axis)
        return uniq

    output = np.unique(consolidated, return_index,
                       return_inverse, return_counts)
    if not (return_index or return_inverse or return_counts):
        return reshape_uniq(output)
    else:
        uniq = reshape_uniq(output[0])
        return tuple([uniq] + list(output[1:]))
